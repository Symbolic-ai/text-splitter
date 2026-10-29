/*!
# [`MarkdownSplitter`]
Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, according to the Common Mark specification.
*/

use std::{iter::once, ops::Range};

use either::Either;
use itertools::Itertools;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use crate::{
    splitter::{fallback::FallbackLevel, SemanticLevel, Splitter},
    trim::Trim,
    ChunkConfig, ChunkSizer,
};

use super::ChunkCharIndex;

/// Markdown splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will
/// attempt to merge neighboring chunks if they can fit within the
/// given chunk size.
#[derive(Debug)]
pub struct MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_config: ChunkConfig<Sizer>,
}

impl<Sizer> MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Creates a new [`MarkdownSplitter`].
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// // By default, the chunk sizer is based on characters.
    /// let splitter = MarkdownSplitter::new(512);
    /// ```
    #[must_use]
    pub fn new(chunk_config: impl Into<ChunkConfig<Sizer>>) -> Self {
        Self {
            chunk_config: chunk_config.into(),
        }
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:
    ///
    /// 1. Characters
    /// 2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
    /// 6. Inline elements such as: text nodes, emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
    /// 7. Sentences of prose in a paragraph, list item, block quote or footnote. A chunk goes below a sentence only when the sentence does not fit.
    /// 8. Block elements suce as: paragraphs, code blocks, footnote definitions, metadata. Also, a block quote or row/item within a table or list that can contain other "block" type elements, and a list or table that contains items.
    /// 9. Thematic breaks or horizontal rules.
    /// 10. Headings by level
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.
    ///
    /// Markdown is parsed according to the Commonmark spec, along with some optional features such as GitHub Flavored Markdown.
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["# Header", "from a", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        Splitter::<_>::chunks(self, text)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "# Header"), (10, "from a"), (17, "document")], chunks);
    /// ```
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        Splitter::<_>::chunk_indices(self, text)
    }

    /// Returns an iterator over chunks of the text with their byte and character offsets.
    /// Each chunk will be up to the `chunk_capacity`.
    ///
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// This will be more expensive than just byte offsets, and for most usage in Rust, just
    /// having byte offsets is sufficient. But when interfacing with other languages or systems
    /// that require character offsets, this will track the character offsets for you,
    /// accounting for any trimming that may have occurred.
    ///
    /// ```
    /// use text_splitter::{ChunkCharIndex, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_char_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![ChunkCharIndex { chunk: "# Header", byte_offset: 0, char_offset: 0 }, ChunkCharIndex { chunk: "from a", byte_offset: 10, char_offset: 10 }, ChunkCharIndex { chunk: "document", byte_offset: 17, char_offset: 17 }], chunks);
    /// ```
    pub fn chunk_char_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = ChunkCharIndex<'text>> + 'splitter {
        Splitter::<_>::chunk_char_indices(self, text)
    }
}

impl<Sizer> Splitter<Sizer> for MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    type Level = Element;

    const TRIM: Trim = Trim::PreserveIndentation;

    fn chunk_config(&self) -> &ChunkConfig<Sizer> {
        &self.chunk_config
    }

    fn parse(&self, text: &str) -> Vec<(Self::Level, Range<usize>)> {
        let mut ranges = Vec::new();
        // Inline content of the paragraph or list item that is open now.
        let mut prose = Prose::default();
        // Index of the `Inline` entry of the previous event, when that event
        // was a text event.
        let mut last_text: Option<usize> = None;
        // Number of code blocks that are open now.
        let mut code_blocks = 0_usize;
        // Container tags that are open now. Sentences are found only in prose.
        let mut containers: Vec<bool> = Vec::new();
        // The outermost inline element of the current line of prose, and the
        // last visible character in it.
        let mut word = Word::default();

        let events = Parser::new_ext(text, Options::all())
            .into_offset_iter()
            .collect::<Vec<_>>();
        for (index, (event, mut range)) in events.iter().cloned().enumerate() {
            if is_inline(&event) {
                if containers.last().copied().unwrap_or(false) {
                    let lead = prose.lead_start();
                    prose.push(text, &event, range.clone());
                    // The first inline element of a paragraph or a list item
                    // starts at the container, so a list marker is never a
                    // section of its own.
                    if let Some(start) = lead {
                        range.start = range.start.min(start);
                    }
                }
            } else if !(opens_lead_container(&event) && prose.is_empty()) {
                // A paragraph or a block quote that opens a list item, a block
                // quote or a footnote definition keeps the outer start.
                prose.flush(&mut ranges);
            }

            match event {
                // pulldown-cmark emits one text node as several events, for
                // example at a smart quote, an entity or an escape. The
                // boundary between them is not semantic: it can fall inside
                // a word ("isn" | "'" | "t"), so join consecutive text events.
                // The range also covers an escape backslash between them.
                // In a code block, each line is a text event: keep the lines.
                Event::Text(_) => {
                    if let Some(index) = last_text.filter(|_| code_blocks == 0) {
                        ranges[index].1.end = range.end;
                        word.enter(&event);
                        continue;
                    }
                    if code_blocks == 0 && word.joins(&ranges, text, &events[index..], &range) {
                        last_text = word.outer;
                        word.extend(&mut ranges, &event, range.end);
                        continue;
                    }
                    last_text = Some(ranges.len());
                    word.start(&ranges, &event);
                    ranges.push((Element::Inline, range));
                    continue;
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    code_blocks += 1;
                    containers.push(false);
                }
                Event::End(TagEnd::CodeBlock) => {
                    code_blocks -= 1;
                    containers.pop();
                }
                Event::Start(ref tag) if !is_inline_tag(tag) => {
                    if matches!(
                        tag,
                        Tag::Paragraph
                            | Tag::Item
                            | Tag::BlockQuote(_)
                            | Tag::FootnoteDefinition(_)
                    ) {
                        prose.open(range.start);
                    }
                    containers.push(matches!(tag, Tag::Paragraph | Tag::Item));
                }
                Event::End(tag) if !is_inline_tag_end(tag) => {
                    containers.pop();
                }
                _ => {}
            }
            last_text = None;

            // An inline element directly next to a word joins the outermost
            // element before it, so a chunk does not end inside the word
            // ("`foo`s", "foo*bar*").
            match event {
                Event::Code(_) | Event::InlineMath(_) | Event::Start(_) if is_inline(&event) => {
                    if word.joins(&ranges, text, &events[index..], &range) {
                        word.extend(&mut ranges, &event, range.end);
                        continue;
                    }
                    word.start(&ranges, &event);
                }
                Event::End(tag) if is_inline_tag_end(tag) => word.close(),
                _ if is_inline(&event) => word.last = None,
                _ => word = Word::default(),
            }

            if let Some(level) = element_for_event(&event) {
                ranges.push((level, range));
            }
        }
        prose.flush(&mut ranges);

        ranges
    }
}

/// The inline elements of the current line of prose, to find the elements
/// that are parts of one word.
#[derive(Debug, Default)]
struct Word {
    /// Index in the ranges of the last inline element that is not inside a
    /// span (emphasis, link, ...).
    outer: Option<usize>,
    /// True when the last event of the outer element is a text event.
    outer_ends_in_text: bool,
    /// Number of inline spans that are open now.
    depth: usize,
    /// Last visible character of the line, when the last inline event shows
    /// text.
    last: Option<char>,
}

impl Word {
    /// True when the inline event at the start of `events` is a part of the
    /// same word as the outer element: it begins directly after the element,
    /// the characters on both sides of the boundary are word characters, and
    /// plain text is on one side ("`foo`s", "foo*bar*"). Two links or spans
    /// next to each other stay separate elements.
    fn joins(
        &self,
        ranges: &[(Element, Range<usize>)],
        text: &str,
        events: &[(Event<'_>, Range<usize>)],
        range: &Range<usize>,
    ) -> bool {
        let Some(outer) = self.outer.filter(|_| self.depth == 0) else {
            return false;
        };
        (self.outer_ends_in_text || matches!(events.first(), Some((Event::Text(_), _))))
            && self.last.is_some_and(is_word_char)
            && first_visible_char(events).is_some_and(is_word_char)
            && text
                .get(ranges[outer].1.end..range.start)
                .is_some_and(str::is_empty)
    }

    /// Extend the outer element to the end of the event that joins it.
    fn extend(&mut self, ranges: &mut [(Element, Range<usize>)], event: &Event<'_>, end: usize) {
        if let Some(outer) = self.outer {
            ranges[outer].1.end = ranges[outer].1.end.max(end);
        }
        self.enter(event);
    }

    /// Record the event as a new element. Its range is the next range.
    fn start(&mut self, ranges: &[(Element, Range<usize>)], event: &Event<'_>) {
        if self.depth == 0 {
            self.outer = Some(ranges.len());
        }
        self.enter(event);
    }

    fn enter(&mut self, event: &Event<'_>) {
        if self.depth == 0 {
            self.outer_ends_in_text = matches!(event, Event::Text(_));
        }
        match event {
            Event::Text(content) | Event::Code(content) | Event::InlineMath(content) => {
                if let Some(last) = content.chars().next_back() {
                    self.last = Some(last);
                }
            }
            Event::Start(_) => self.depth += 1,
            _ => self.last = None,
        }
    }

    fn close(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }
}

/// A letter, a digit or an apostrophe.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '\u{2019}'
}

/// The first visible character of the inline element at the start of
/// `events`.
fn first_visible_char(events: &[(Event<'_>, Range<usize>)]) -> Option<char> {
    events.iter().find_map(|(event, _)| match event {
        Event::Text(content) | Event::Code(content) | Event::InlineMath(content) => {
            Some(content.chars().next())
        }
        Event::Start(tag) if is_inline_tag(tag) => None,
        _ => Some(None),
    })?
}

/// One inline event of a run of prose.
#[derive(Debug)]
struct ProsePiece {
    /// Byte range of the piece in `Prose::visible`.
    visible: Range<usize>,
    /// Byte range of the event in the source text.
    source: Range<usize>,
    /// True when the visible bytes are the source bytes, so an offset in the
    /// piece maps to the source. A sentence never ends inside other pieces.
    exact: bool,
    /// True inside a link, an image or an emphasis span. A sentence never ends
    /// there: a chunk would cut the span in two and leave its syntax open.
    in_span: bool,
}

/// The visible text of a run of prose, which is the input to sentence
/// segmentation. Markdown syntax is not in it: a `!` of an image or a `?` in a
/// link destination is not the end of a sentence.
#[derive(Debug, Default)]
struct Prose {
    /// Start of the paragraph, list item, block quote or footnote definition
    /// that holds the next run. The first sentence starts there, so a list,
    /// task, quote or footnote marker stays with its sentence.
    container_start: Option<usize>,
    /// Number of links, images and emphasis spans that are open now.
    span_depth: usize,
    /// Source range of the run, from its first to its last inline event.
    source: Option<Range<usize>>,
    visible: String,
    pieces: Vec<ProsePiece>,
}

impl Prose {
    /// The container start, when the next inline event is the first of its run.
    fn lead_start(&self) -> Option<usize> {
        self.container_start.filter(|_| self.is_empty())
    }

    fn is_empty(&self) -> bool {
        self.source.is_none()
    }

    fn open(&mut self, start: usize) {
        if self.source.is_none() {
            self.container_start.get_or_insert(start);
        }
    }

    fn push(&mut self, text: &str, event: &Event<'_>, range: Range<usize>) {
        self.source = Some(
            self.source
                .take()
                .map_or(range.clone(), |run| run.start..range.end.max(run.end)),
        );

        let (visible, exact) = match event {
            Event::Text(content) => (content.as_ref(), text.get(range.clone()) == Some(content)),
            Event::Code(content) => (content.as_ref(), false),
            // A footnote label, inline HTML or math is not prose. Its content
            // could hide a sentence boundary next to it ("text.[^1] Next").
            // Markdown joins the lines of a paragraph with a space.
            Event::InlineHtml(_)
            | Event::InlineMath(_)
            | Event::FootnoteReference(_)
            | Event::SoftBreak
            | Event::HardBreak => (" ", false),
            Event::Start(_) => {
                self.span_depth += 1;
                return;
            }
            Event::End(_) => {
                self.span_depth = self.span_depth.saturating_sub(1);
                return;
            }
            _ => return,
        };
        let start = self.visible.len();
        self.visible.push_str(visible);
        self.pieces.push(ProsePiece {
            visible: start..self.visible.len(),
            source: range,
            exact,
            in_span: self.span_depth > 0,
        });
    }

    /// Add a `Sentence` range for each sentence of the run, when the run holds
    /// more than one sentence. Then start a new run.
    fn flush(&mut self, ranges: &mut Vec<(Element, Range<usize>)>) {
        let container_start = self.container_start.take();
        let Some(run) = self.source.take() else {
            return;
        };

        let boundaries = FallbackLevel::Sentence
            .sections(&self.visible)
            .skip(1)
            .filter_map(|(offset, _)| self.source_offset(offset))
            .filter(|offset| run.start < *offset && *offset < run.end)
            .collect::<Vec<_>>();

        if !boundaries.is_empty() {
            let mut start = container_start.map_or(run.start, |start| start.min(run.start));
            for end in boundaries.into_iter().chain(once(run.end)) {
                if start < end {
                    ranges.push((Element::Sentence, start..end));
                    start = end;
                }
            }
        }

        self.visible.clear();
        self.pieces.clear();
        self.span_depth = 0;
    }

    /// Map a sentence boundary in the visible text to the source text. A
    /// boundary between two pieces goes after the first piece, so the syntax
    /// that opens the next piece (`[`, `*`, ...) stays with the next sentence.
    fn source_offset(&self, offset: usize) -> Option<usize> {
        let index = self
            .pieces
            .partition_point(|piece| piece.visible.end < offset);
        let piece = self.pieces.get(index).filter(|piece| !piece.in_span)?;
        if piece.visible.end == offset {
            Some(piece.source.end)
        } else if piece.exact && piece.visible.start < offset {
            Some(piece.source.start + offset - piece.visible.start)
        } else {
            None
        }
    }
}

/// Map a parser event to the semantic level of its range.
fn element_for_event(event: &Event<'_>) -> Option<Element> {
    match event {
        Event::Start(
            Tag::Emphasis
            | Tag::Strong
            | Tag::Strikethrough
            | Tag::Link { .. }
            | Tag::Image { .. }
            | Tag::Subscript
            | Tag::Superscript
            | Tag::TableCell,
        )
        | Event::Text(_)
        | Event::HardBreak
        | Event::Code(_)
        | Event::InlineHtml(_)
        | Event::InlineMath(_)
        | Event::FootnoteReference(_)
        | Event::TaskListMarker(_) => Some(Element::Inline),
        Event::SoftBreak => Some(Element::SoftBreak),
        Event::Html(_)
        | Event::DisplayMath(_)
        | Event::Start(
            Tag::Paragraph
            | Tag::CodeBlock(_)
            | Tag::FootnoteDefinition(_)
            | Tag::MetadataBlock(_)
            | Tag::TableHead
            | Tag::BlockQuote(_)
            | Tag::TableRow
            | Tag::Item
            | Tag::HtmlBlock
            | Tag::List(_)
            | Tag::Table(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition,
        ) => Some(Element::Block),
        Event::Rule => Some(Element::Rule),
        Event::Start(Tag::Heading { level, .. }) => Some(Element::Heading((*level).into())),
        Event::End(_) => None,
    }
}

/// True for a start event that opens a container whose first sentence keeps
/// the container start (see `Prose::container_start`).
fn opens_lead_container(event: &Event<'_>) -> bool {
    matches!(
        event,
        Event::Start(Tag::Paragraph | Tag::BlockQuote(_) | Tag::FootnoteDefinition(_))
    )
}

// The inline classification lives in four functions that must agree:
// `element_for_event` (the level of each range), `is_inline_tag` and
// `is_inline_tag_end` (spans inside a line of prose), and `is_inline` (the
// events of a run of prose). Two differences are deliberate. `TableCell` is
// an `Inline` level but not prose: a cell is not part of a sentence. A task
// list marker is prose, so it stays with the first sentence of its item.

/// True for the tags that stay inside a line of prose.
fn is_inline_tag(tag: &Tag<'_>) -> bool {
    matches!(
        tag,
        Tag::Emphasis
            | Tag::Strong
            | Tag::Strikethrough
            | Tag::Link { .. }
            | Tag::Image { .. }
            | Tag::Subscript
            | Tag::Superscript
    )
}

/// True for the end of a tag that stays inside a line of prose.
fn is_inline_tag_end(tag: TagEnd) -> bool {
    matches!(
        tag,
        TagEnd::Emphasis
            | TagEnd::Strong
            | TagEnd::Strikethrough
            | TagEnd::Link
            | TagEnd::Image
            | TagEnd::Subscript
            | TagEnd::Superscript
    )
}

/// True for an event that is part of a run of prose.
fn is_inline(event: &Event<'_>) -> bool {
    match event {
        Event::Text(_)
        | Event::Code(_)
        | Event::InlineHtml(_)
        | Event::InlineMath(_)
        | Event::FootnoteReference(_)
        | Event::SoftBreak
        | Event::HardBreak
        | Event::TaskListMarker(_) => true,
        Event::Start(tag) => is_inline_tag(tag),
        Event::End(tag) => is_inline_tag_end(*tag),
        _ => false,
    }
}

/// Heading levels in markdown.
/// Sorted in reverse order for sorting purposes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum HeadingLevel {
    H6,
    H5,
    H4,
    H3,
    H2,
    H1,
}

impl From<pulldown_cmark::HeadingLevel> for HeadingLevel {
    fn from(value: pulldown_cmark::HeadingLevel) -> Self {
        match value {
            pulldown_cmark::HeadingLevel::H1 => HeadingLevel::H1,
            pulldown_cmark::HeadingLevel::H2 => HeadingLevel::H2,
            pulldown_cmark::HeadingLevel::H3 => HeadingLevel::H3,
            pulldown_cmark::HeadingLevel::H4 => HeadingLevel::H4,
            pulldown_cmark::HeadingLevel::H5 => HeadingLevel::H5,
            pulldown_cmark::HeadingLevel::H6 => HeadingLevel::H6,
        }
    }
}

/// How a particular semantic level relates to surrounding text elements.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SemanticSplitPosition {
    /// The semantic level should be treated as its own chunk.
    Own,
    /// The semantic level should be included in the next chunk.
    Next,
}

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Element {
    /// Single line break, which isn't necessarily a new element in Markdown
    SoftBreak,
    /// An inline element that is within a larger element such as a paragraph, but
    /// more specific than a sentence.
    Inline,
    /// A sentence of prose in a paragraph or a list item. It ranks above the
    /// inline elements, so a chunk ends at a sentence boundary when the
    /// sentence fits. Inline elements split only a sentence that is too large.
    Sentence,
    /// Paragraph, code block, metadata, a row/item within a table or list, block quote, that can contain other "block" type elements, List or table that contains items
    Block,
    /// thematic break/horizontal rule
    Rule,
    /// Heading levels in markdown
    Heading(HeadingLevel),
}

impl Element {
    fn split_position(self) -> SemanticSplitPosition {
        match self {
            Self::SoftBreak | Self::Block | Self::Rule | Self::Inline | Self::Sentence => {
                SemanticSplitPosition::Own
            }
            // Attach it to the next text
            Self::Heading(_) => SemanticSplitPosition::Next,
        }
    }

    fn treat_whitespace_as_previous(self) -> bool {
        match self {
            Self::SoftBreak | Self::Inline | Self::Sentence | Self::Rule | Self::Heading(_) => {
                false
            }
            Self::Block => true,
        }
    }
}

impl SemanticLevel for Element {
    fn sections(
        text: &str,
        level_ranges: impl Iterator<Item = (Self, Range<usize>)>,
    ) -> impl Iterator<Item = (usize, &str)> {
        let mut cursor = 0;
        let mut final_match = false;
        level_ranges
            .batching(move |it| {
                loop {
                    match it.next() {
                        // If we've hit the end, actually return None
                        None if final_match => return None,
                        // First time we hit None, return the final section of the text
                        None => {
                            final_match = true;
                            return text.get(cursor..).map(|t| Either::Left(once((cursor, t))));
                        }
                        // Return text preceding match + the match
                        Some((level, range)) => {
                            let offset = cursor;
                            match level.split_position() {
                                SemanticSplitPosition::Own => {
                                    if range.start < cursor {
                                        continue;
                                    }
                                    let prev_section = text
                                        .get(cursor..range.start)
                                        .expect("invalid character sequence");
                                    if level.treat_whitespace_as_previous()
                                        && prev_section.chars().all(char::is_whitespace)
                                    {
                                        let section = text
                                            .get(cursor..range.end)
                                            .expect("invalid character sequence");
                                        cursor = range.end;
                                        return Some(Either::Left(once((offset, section))));
                                    }
                                    let separator = text
                                        .get(range.start..range.end)
                                        .expect("invalid character sequence");
                                    cursor = range.end;
                                    return Some(Either::Right(
                                        [(offset, prev_section), (range.start, separator)]
                                            .into_iter(),
                                    ));
                                }
                                SemanticSplitPosition::Next => {
                                    if range.start < cursor {
                                        continue;
                                    }
                                    let prev_section = text
                                        .get(cursor..range.start)
                                        .expect("invalid character sequence");
                                    // Separator will be part of the next chunk
                                    cursor = range.start;
                                    return Some(Either::Left(once((offset, prev_section))));
                                }
                            }
                        }
                    }
                }
            })
            .flatten()
            .filter(|(_, s)| !s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use crate::splitter::SemanticSplitRanges;

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let chunks = MarkdownSplitter::new(ChunkConfig::new(text.chars().count()).with_trim(false))
            .chunks(&text)
            .collect::<Vec<_>>();

        assert_eq!(vec![&text], chunks);
    }

    #[test]
    fn returns_two_chunks_if_text_is_longer_than_max_chunk_size() {
        let text1 = Faker.fake::<String>();
        let text2 = Faker.fake::<String>();
        let text = format!("{text1}{text2}");
        // Round up to one above half so it goes to 2 chunks
        let max_chunk_size = text.chars().count() / 2 + 1;

        let chunks = MarkdownSplitter::new(ChunkConfig::new(max_chunk_size).with_trim(false))
            .chunks(&text)
            .collect::<Vec<_>>();

        assert!(chunks.iter().all(|c| c.chars().count() <= max_chunk_size));

        // Check that beginning of first chunk and text 1 matches
        let len = min(text1.len(), chunks[0].len());
        assert_eq!(text1[..len], chunks[0][..len]);
        // Check that end of second chunk and text 2 matches
        let len = min(text2.len(), chunks[1].len());
        assert_eq!(
            text2[(text2.len() - len)..],
            chunks[1][chunks[1].len() - len..]
        );

        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn empty_string() {
        let text = "";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(100).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks = MarkdownSplitter::new(ChunkConfig::new(1).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(3).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";
        let chunks = MarkdownSplitter::new(1)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn chunk_char_indices() {
        let text = " a b ";
        let chunks = MarkdownSplitter::new(1)
            .chunk_char_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                ChunkCharIndex {
                    chunk: "a",
                    byte_offset: 1,
                    char_offset: 1
                },
                ChunkCharIndex {
                    chunk: "b",
                    byte_offset: 3,
                    char_offset: 3,
                },
            ],
            chunks
        );
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(1).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
            chunks
        );
    }

    #[test]
    fn trim_grapheme_indices() {
        let text = "\r\na̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(3)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn grapheme_char_indices() {
        let text = "\r\na̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(3)
            .chunk_char_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                ChunkCharIndex {
                    chunk: "a̐é",
                    byte_offset: 2,
                    char_offset: 2
                },
                ChunkCharIndex {
                    chunk: "ö̲",
                    byte_offset: 7,
                    char_offset: 5
                }
            ],
            chunks
        );
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick brown fox can jump 32.3 feet, right?";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(10).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                "The quick ",
                "brown fox ",
                "can jump ",
                "32.3 feet,",
                " right?"
            ],
            chunks
        );
    }

    #[test]
    fn words_fallback_to_graphemes() {
        let text = "Thé quick\r\n";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(2).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks = MarkdownSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(21).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["Mr. Fox jumped. ", "The dog was too lazy."], chunks);
    }

    #[test]
    fn sentences_falls_back_to_words() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(16).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec!["Mr. Fox jumped. ", "The dog was too ", "lazy."],
            chunks
        );
    }

    #[test]
    fn trim_sentence_indices() {
        let text = "Some text. From a document.";
        let chunks = MarkdownSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn adjacent_text_events_are_one_inline_range() {
        // Smart punctuation, an entity and an escape each split a text node.
        let text = "It isn't &amp; \\*here\\*.";
        let ranges = MarkdownSplitter::new(10).parse(text);

        assert_eq!(
            vec![
                (Element::Block, 0..text.len()),
                (Element::Inline, 0..text.len())
            ],
            ranges
        );
    }

    #[test]
    fn sentences_of_a_paragraph_are_a_level() {
        let text = "One isn't two. [See *this*](https://a.example/?q=Yes!No) now! Three.";
        let markdown = SemanticSplitRanges::new(MarkdownSplitter::new(10).parse(text));
        let sentences = markdown
            .ranges_after_offset(0)
            .filter(|(level, _)| *level == Element::Sentence)
            .map(|(_, range)| &text[range])
            .collect::<Vec<_>>();

        // The `?` and `!` in the link destination do not end a sentence.
        assert_eq!(
            vec![
                "One isn't two. ",
                "[See *this*](https://a.example/?q=Yes!No) now! ",
                "Three."
            ],
            sentences
        );
    }

    #[test]
    fn sentences_join_the_lines_of_a_paragraph() {
        let text = "One sentence that\nwraps. Two.";
        let markdown = SemanticSplitRanges::new(MarkdownSplitter::new(10).parse(text));
        let sentences = markdown
            .ranges_after_offset(0)
            .filter(|(level, _)| *level == Element::Sentence)
            .map(|(_, range)| &text[range])
            .collect::<Vec<_>>();

        assert_eq!(vec!["One sentence that\nwraps. ", "Two."], sentences);
    }

    #[test]
    fn code_block_lines_stay_separate_and_have_no_sentences() {
        // In a list item, pulldown-cmark emits each code line as a text event.
        let text = "- item\n\n  ```\n  One. Two.\n  Three.\n  ```";
        let ranges = MarkdownSplitter::new(10).parse(text);
        let code_lines = ranges
            .iter()
            .filter(|(level, range)| *level == Element::Inline && range.start > 8)
            .map(|(_, range)| &text[range.clone()])
            .collect::<Vec<_>>();

        assert_eq!(vec!["One. Two.\n", "Three.\n"], code_lines);
        assert!(ranges
            .iter()
            .all(|(level, range)| *level != Element::Sentence || range.end <= 8));
    }

    fn sentences(text: &str) -> Vec<&str> {
        let markdown = SemanticSplitRanges::new(MarkdownSplitter::new(10).parse(text));
        markdown
            .ranges_after_offset(0)
            .filter(|(level, _)| *level == Element::Sentence)
            .map(|(_, range)| &text[range])
            .collect()
    }

    #[test]
    fn sentence_never_ends_inside_a_link_image_or_emphasis() {
        assert_eq!(
            vec!["One. ", "[Here. Now](http://x) more. ", "Two."],
            sentences("One. [Here. Now](http://x) more. Two.")
        );
        assert_eq!(
            vec!["One. ", "![Figure 1. Overview](x) shows it. ", "Two."],
            sentences("One. ![Figure 1. Overview](x) shows it. Two.")
        );
        assert_eq!(
            vec!["One. ", "**Note. Next** words. ", "Two."],
            sentences("One. **Note. Next** words. Two.")
        );
    }

    #[test]
    fn sentence_never_ends_inside_a_code_span() {
        assert_eq!(
            vec!["Run `a. B` now. ", "Two."],
            sentences("Run `a. B` now. Two.")
        );
    }

    #[test]
    fn footnote_reference_and_inline_html_keep_the_sentence_boundary() {
        assert_eq!(
            vec!["One.[^1] ", "Two."],
            sentences("One.[^1] Two.\n\n[^1]: Note")
        );
        assert_eq!(vec!["One.<br> ", "Two."], sentences("One.<br> Two."));
    }

    #[test]
    fn chunk_ends_at_a_sentence_when_the_next_sentence_does_not_fit() {
        let text = "It isn't far. It certainly isn't visible.";
        let chunks = MarkdownSplitter::new(30).chunks(text).collect::<Vec<_>>();

        assert_eq!(vec!["It isn't far.", "It certainly isn't visible."], chunks);
    }

    #[test]
    fn sentence_level_keeps_inline_markup_whole() {
        let text = "Read [the *full* report](https://a.example/r). It is long enough.";
        let chunks = MarkdownSplitter::new(50).chunks(text).collect::<Vec<_>>();

        assert_eq!(
            vec![
                "Read [the *full* report](https://a.example/r).",
                "It is long enough."
            ],
            chunks
        );
    }

    #[test]
    fn list_marker_stays_with_its_text() {
        let text = "- First item text that goes on\n- Second";
        let chunks = MarkdownSplitter::new(24).chunks(text).collect::<Vec<_>>();

        assert_eq!(vec!["- First item text that", "goes on\n- Second"], chunks);
    }

    #[test]
    fn marker_stays_with_the_first_sentence() {
        for (text, first) in [
            (
                "1. First sentence here. Second one.",
                "1. First sentence here. ",
            ),
            (
                "- First sentence here. Second one.",
                "- First sentence here. ",
            ),
            (
                "- [ ] First sentence here. Second one.",
                "- [ ] First sentence here. ",
            ),
            (
                "> First sentence here. Second one.",
                "> First sentence here. ",
            ),
        ] {
            assert_eq!(vec![first, "Second one."], sentences(text), "{text}");
        }
    }

    #[test]
    fn ordered_list_marker_stays_with_the_first_words() {
        let text = "1. Actual numbers don't matter, just that it's a number";
        let chunks = MarkdownSplitter::new(32).chunks(text).collect::<Vec<_>>();

        assert_eq!(
            vec!["1. Actual numbers don't matter,", "just that it's a number"],
            chunks
        );
    }

    fn inline_sections(text: &str) -> Vec<&str> {
        SemanticSplitRanges::new(MarkdownSplitter::new(10).parse(text))
            .semantic_chunks(0, text, Element::Inline)
            .map(|(_, section)| section)
            .collect()
    }

    #[test]
    fn inline_element_next_to_a_word_is_not_a_boundary() {
        for (text, expected) in [
            ("Call `foo`s now", vec!["Call ", "`foo`s now"]),
            ("Call *foo*s now", vec!["Call ", "*foo*s now"]),
            ("Call [foo](u)s now", vec!["Call ", "[foo](u)s now"]),
            ("Call foo`bar` now", vec!["Call foo`bar`", " now"]),
            ("Call foo*bar* now", vec!["Call foo*bar*", " now"]),
            ("Call `foo`'s now", vec!["Call ", "`foo`'s now"]),
        ] {
            assert_eq!(expected, inline_sections(text), "{text}");
        }
    }

    #[test]
    fn inline_element_next_to_a_space_or_a_span_is_still_a_boundary() {
        for (text, expected) in [
            ("a `foo` b", vec!["a ", "`foo`", " b"]),
            ("a *foo* b", vec!["a ", "*foo*", " b"]),
            ("text<br>more", vec!["text", "<br>", "more"]),
            ("Call `foo`, now", vec!["Call ", "`foo`", ", now"]),
            ("[a](u)[b](v) c", vec!["[a](u)", "[b](v)", " c"]),
            ("**a**_b_ c", vec!["**a**", "_b_", " c"]),
        ] {
            assert_eq!(expected, inline_sections(text), "{text}");
        }
    }

    #[test]
    fn chunk_does_not_end_between_a_code_span_and_its_suffix() {
        let text = "Some words before `foo`s and some words after it.";
        let chunks = MarkdownSplitter::new(24).chunks(text).collect::<Vec<_>>();

        assert!(
            chunks.iter().all(|chunk| !chunk.starts_with('s')),
            "{chunks:?}"
        );
    }

    #[test]
    fn test_no_markdown_separators() {
        let splitter = MarkdownSplitter::new(10);
        let markdown =
            SemanticSplitRanges::new(splitter.parse("Some text without any markdown separators"));

        assert_eq!(
            vec![(Element::Block, 0..41), (Element::Inline, 0..41)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_checklist() {
        let splitter = MarkdownSplitter::new(10);
        let markdown =
            SemanticSplitRanges::new(splitter.parse("- [ ] incomplete task\n- [x] completed task"));

        assert_eq!(
            vec![
                (Element::Block, 0..42),
                (Element::Block, 0..22),
                (Element::Inline, 0..5),
                (Element::Inline, 6..21),
                (Element::Block, 22..42),
                (Element::Inline, 22..27),
                (Element::Inline, 28..42),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_reference() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Footnote[^1]"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..8),
                (Element::Inline, 8..12),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_code() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("`bash`"));

        assert_eq!(
            vec![(Element::Block, 0..6), (Element::Inline, 0..6)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_emphasis() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("*emphasis*"));

        assert_eq!(
            vec![
                (Element::Block, 0..10),
                (Element::Inline, 0..10),
                (Element::Inline, 1..9),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strong() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("**emphasis**"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..12),
                (Element::Inline, 2..10),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strikethrough() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("~~emphasis~~"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..12),
                (Element::Inline, 2..10),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_link() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("[link](url)"));

        assert_eq!(
            vec![
                (Element::Block, 0..11),
                (Element::Inline, 0..11),
                (Element::Inline, 1..5),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("![link](url)"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..12),
                (Element::Inline, 2..6),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_html() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("<span>Some text</span>"));

        assert_eq!(
            vec![
                (Element::Block, 0..22),
                (Element::Inline, 0..6),
                (Element::Inline, 6..15),
                (Element::Inline, 15..22),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_html() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("<div>Some text</div>"));

        assert_eq!(
            vec![(Element::Block, 0..20), (Element::Block, 0..20)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_table() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(
            splitter.parse("| Header 1 | Header 2 |\n| --- | --- |\n| Cell 1 | Cell 2 |"),
        );
        assert_eq!(
            vec![
                (Element::Block, 0..57),
                (Element::Block, 0..24),
                (Element::Inline, 1..11),
                (Element::Inline, 2..10),
                (Element::Inline, 12..22),
                (Element::Inline, 13..21),
                (Element::Block, 38..57),
                (Element::Inline, 39..47),
                (Element::Inline, 40..46),
                (Element::Inline, 48..56),
                (Element::Inline, 49..55)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_softbreak() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Some text\nwith a softbreak"));

        assert_eq!(
            vec![
                (Element::Block, 0..26),
                (Element::Inline, 0..9),
                (Element::SoftBreak, 9..10),
                (Element::Inline, 10..26)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hardbreak() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Some text\\\nwith a hardbreak"));

        assert_eq!(
            vec![
                (Element::Block, 0..27),
                (Element::Inline, 0..9),
                (Element::Inline, 9..11),
                (Element::Inline, 11..27)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_def() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("[^first]: Footnote"));

        assert_eq!(
            vec![
                (Element::Block, 0..18),
                (Element::Inline, 0..18),
                (Element::Block, 10..18)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_code_block() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("```\ncode\n```"));

        assert_eq!(
            vec![(Element::Block, 0..12), (Element::Inline, 4..9)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_block_quote() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("> quote"));

        assert_eq!(
            vec![
                (Element::Block, 0..7),
                (Element::Inline, 0..7),
                (Element::Block, 2..7)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_with_rule() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Some text\n\n---\n\nwith a rule"));

        assert_eq!(
            vec![
                (Element::Block, 0..10),
                (Element::Inline, 0..9),
                (Element::Rule, 11..15),
                (Element::Block, 16..27),
                (Element::Inline, 16..27)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_heading() {
        for (index, (heading, level)) in [
            ("#", HeadingLevel::H1),
            ("##", HeadingLevel::H2),
            ("###", HeadingLevel::H3),
            ("####", HeadingLevel::H4),
            ("#####", HeadingLevel::H5),
            ("######", HeadingLevel::H6),
        ]
        .into_iter()
        .enumerate()
        {
            let splitter = MarkdownSplitter::new(10);
            let markdown = SemanticSplitRanges::new(splitter.parse(&format!("{heading} Heading")));

            assert_eq!(
                vec![
                    (Element::Heading(level), 0..9 + index),
                    (Element::Inline, 2 + index..9 + index)
                ],
                markdown.ranges_after_offset(0).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_ranges_after_offset_block() {
        let splitter = MarkdownSplitter::new(10);
        let markdown =
            SemanticSplitRanges::new(splitter.parse("- [ ] incomplete task\n- [x] completed task"));

        assert_eq!(
            vec![(Element::Block, 0..22), (Element::Block, 22..42),],
            markdown
                .level_ranges_after_offset(0, Element::Block)
                .collect::<Vec<_>>()
        );
    }
}

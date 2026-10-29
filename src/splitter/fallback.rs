use std::sync::LazyLock;

use auto_enums::auto_enum;
use icu_segmenter::{
    options::{SentenceBreakInvariantOptions, WordBreakInvariantOptions},
    GraphemeClusterSegmenter, GraphemeClusterSegmenterBorrowed, SentenceSegmenter,
    SentenceSegmenterBorrowed, WordSegmenter, WordSegmenterBorrowed,
};
use itertools::Itertools;
use strum::EnumIter;

pub const GRAPHEME_SEGMENTER: GraphemeClusterSegmenterBorrowed<'static> =
    GraphemeClusterSegmenter::new();
static WORD_SEGMENTER: LazyLock<WordSegmenterBorrowed<'static>> =
    LazyLock::new(|| WordSegmenter::new_dictionary(WordBreakInvariantOptions::default()));
static SENTENCE_SEGMENTER: LazyLock<SentenceSegmenterBorrowed<'static>> =
    LazyLock::new(|| SentenceSegmenter::new(SentenceBreakInvariantOptions::default()));

/// When using a custom semantic level, it is possible that none of them will
/// be small enough to fit into the chunk size. In order to make sure we can
/// still move the cursor forward, we fallback to unicode segmentation.
#[derive(Clone, Copy, Debug, EnumIter, Eq, PartialEq, Ord, PartialOrd)]
pub enum FallbackLevel {
    /// Split by individual chars. May be larger than a single byte,
    /// but we don't go lower so we always have valid UTF str's.
    Char,
    /// Split by [unicode grapheme clusters](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)    Grapheme,
    GraphemeCluster,
    /// Split by [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    Word,
    /// Split by [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    Sentence,
}

impl FallbackLevel {
    /// The next smaller boundary used to probe a long section.
    pub fn boundary_level_for_probe(self) -> Option<Self> {
        match self {
            Self::Sentence => Some(Self::Word),
            Self::Char | Self::GraphemeCluster | Self::Word => None,
        }
    }

    /// Find the first section without scanning the whole remaining document.
    /// The returned section is a lower bound when `truncated` is true.
    pub fn first_section_within(self, text: &str, window: usize) -> Option<(&str, bool)> {
        if window >= text.len() {
            return self
                .sections(text)
                .next()
                .map(|(_, section)| (section, false));
        }

        let mut end = window.max(1);
        while !text.is_char_boundary(end) {
            end += 1;
        }

        match self.sections(&text[..end]).next() {
            // A boundary near the cut may disappear with more context.
            Some((_, section)) if section.len() >= end / 2 => Some((section, true)),
            Some((_, section)) if section.len() < end => self
                .sections(text)
                .next()
                .map(|(_, section)| (section, false)),
            _ => Some((&text[..end], true)),
        }
    }

    #[auto_enum(Iterator)]
    pub fn sections(self, text: &str) -> impl Iterator<Item = (usize, &str)> {
        match self {
            Self::Char => text.char_indices().map(move |(i, c)| {
                (
                    i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
            }),
            Self::GraphemeCluster => GRAPHEME_SEGMENTER
                .segment_str(text)
                .tuple_windows()
                .map(|(i, j)| (i, &text[i..j])),
            Self::Word => WORD_SEGMENTER
                .segment_str(text)
                .tuple_windows()
                .map(|(i, j)| (i, &text[i..j])),
            // A list marker ("1. ", "2) ") ends a Unicode sentence when the
            // next word starts with a capital letter. It is not a sentence:
            // it stays with the words after it.
            Self::Sentence => SENTENCE_SEGMENTER
                .segment_str(text)
                .tuple_windows()
                .map(|(i, j)| (i, &text[i..j]))
                .coalesce(|(i, a), (j, b)| {
                    if is_list_marker(a) {
                        Ok((i, &text[i..j + b.len()]))
                    } else {
                        Err(((i, a), (j, b)))
                    }
                }),
        }
    }
}

/// True for a section that holds only an ordered list marker: one to nine
/// digits, then `.` or `)`.
fn is_list_marker(section: &str) -> bool {
    let Some(label) = section
        .trim()
        .strip_suffix('.')
        .or_else(|| section.trim().strip_suffix(')'))
    else {
        return false;
    };
    (1..=9).contains(&label.len()) && label.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn list_marker_is_not_a_sentence() {
        for (text, expected) in [
            (
                "1. Actual numbers don't matter.",
                vec!["1. Actual numbers don't matter."],
            ),
            (
                "12) First item. 3. Second item.",
                vec!["12) First item. ", "3. Second item."],
            ),
            ("E. M. Forster.", vec!["E. ", "M. ", "Forster."]),
            (
                "It ends at 1. Then more.",
                vec!["It ends at 1. ", "Then more."],
            ),
            ("1. ", vec!["1. "]),
        ] {
            let sections = FallbackLevel::Sentence
                .sections(text)
                .map(|(_, section)| section)
                .collect::<Vec<_>>();
            assert_eq!(expected, sections, "{text}");
        }
    }

    #[test]
    fn bounded_first_section_is_a_prefix_of_the_full_section() {
        let text = "1. Short sentence one. Another sentence follows here. 2. And a third.";

        for level in FallbackLevel::iter() {
            let expected = level.sections(text).next().unwrap().1;
            for window in [1, 5, 18, 19, 20, 40, text.len(), text.len() + 10] {
                let (section, truncated) = level.first_section_within(text, window).unwrap();
                assert!(expected.starts_with(section), "{level:?} {window}");
                if !truncated {
                    assert_eq!(section, expected, "{level:?} {window}");
                }
            }
        }
    }

    #[test]
    fn bounded_section_handles_long_unbroken_text_and_unicode() {
        let text = "https://example.com/a 2026-01-01T00:00:00+02:00   ".repeat(2_000);
        let (section, truncated) = FallbackLevel::Sentence
            .first_section_within(&text, 1_000)
            .unwrap();
        assert!(truncated);
        assert!(section.len() <= 1_000);

        let unicode = "é".repeat(1_000);
        for window in 1..10 {
            let (section, _) = FallbackLevel::Sentence
                .first_section_within(&unicode, window)
                .unwrap();
            assert!(unicode.is_char_boundary(section.len()));
        }
    }
}

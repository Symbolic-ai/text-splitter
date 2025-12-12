#![cfg(feature = "bpe")]

use bpe_openai::{cl100k_base, o200k_base};
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

const SYMBOLIC_MARKDOWN: &str = include_str!("inputs/symbolic_markdown.md");

fn char_offset_of(text: &str, byte_offset: usize) -> usize {
    text[..byte_offset].chars().count()
}

#[test]
fn text_splitter_respects_bpe_token_limits() {
    let tokenizer = cl100k_base();
    let splitter = TextSplitter::new(
        ChunkConfig::new(10)
            .with_trim(false)
            .with_sizer(tokenizer),
    );

    let text = "This is a short paragraph with enough words to require \
                multiple chunks when the capacity is limited to ten tokens.";

    let chunk_indices = splitter.chunk_indices(text).collect::<Vec<_>>();
    assert!(!chunk_indices.is_empty());

    let mut previous_end = 0;
    for (offset, chunk) in &chunk_indices {
        assert_eq!(*offset, previous_end, "chunks should be contiguous");
        previous_end = offset + chunk.len();

        let token_count = cl100k_base().count(chunk);
        assert!(
            token_count <= 10,
            "chunk exceeded token capacity: {token_count} tokens in `{chunk}`"
        );
    }

    assert_eq!(previous_end, text.len(), "chunks should cover the full input");

    let reconstructed: String = chunk_indices.iter().map(|(_, chunk)| *chunk).collect();
    assert_eq!(reconstructed, text, "reconstructed text should match input");
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_chunk_char_indices_are_aligned_with_bpe() {
    let tokenizer = cl100k_base();
    let splitter = MarkdownSplitter::new(
        ChunkConfig::new(60)
            .with_trim(false)
            .with_sizer(tokenizer),
    );

    let chunks = splitter
        .chunk_char_indices(SYMBOLIC_MARKDOWN)
        .collect::<Vec<_>>();

    assert!(!chunks.is_empty(), "expected fixture to produce chunks");

    let mut last_byte = 0;
    for chunk in &chunks {
        // Byte offsets should be monotonic and match the slice in the source text.
        assert!(chunk.byte_offset >= last_byte, "byte offsets should increase");
        let source_slice =
            &SYMBOLIC_MARKDOWN[chunk.byte_offset..chunk.byte_offset + chunk.chunk.len()];
        assert_eq!(source_slice, chunk.chunk, "chunk content should match source slice");

        // Char offsets should match the number of chars before the byte offset.
        let expected_char_offset = char_offset_of(SYMBOLIC_MARKDOWN, chunk.byte_offset);
        assert_eq!(
            chunk.char_offset, expected_char_offset,
            "char offset should align with byte offset"
        );

        let token_count = cl100k_base().count(chunk.chunk);
        assert!(
            token_count <= 60,
            "chunk exceeded token capacity: {token_count} tokens in `{}`",
            chunk.chunk
        );

        last_byte = chunk.byte_offset + chunk.chunk.len();
    }
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_splitter_handles_graphemes_under_bpe_limits() {
    let tokenizer = o200k_base();
    let splitter = MarkdownSplitter::new(
        ChunkConfig::new(8)
            .with_trim(false)
            .with_sizer(tokenizer),
    );

    // Includes multi-codepoint graphemes (skin tone + ZWJ sequences).
    let text = "# Emojis\n\n🤦🏽‍♂️🤷🏽‍♀️ keep their boundaries intact.\n";

    let chunk_indices = splitter.chunk_indices(text).collect::<Vec<_>>();
    assert!(!chunk_indices.is_empty());

    let reconstructed: String = chunk_indices.iter().map(|(_, chunk)| *chunk).collect();
    assert_eq!(reconstructed, text, "should preserve grapheme boundaries");

    for (_, chunk) in &chunk_indices {
        let token_count = o200k_base().count(chunk);
        assert!(
            token_count <= 8,
            "chunk exceeded token capacity: {token_count} tokens in `{chunk}`"
        );
    }
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_splitter_bpe_with_default_trim_stays_monotonic() {
    // Default trim is true; ensure offsets still increase and token budgets hold.
    let tokenizer = cl100k_base();
    let splitter = MarkdownSplitter::new(ChunkConfig::new(40).with_sizer(tokenizer));

    let chunk_indices = splitter
        .chunk_indices(SYMBOLIC_MARKDOWN)
        .collect::<Vec<_>>();

    assert!(!chunk_indices.is_empty());

    let mut last_offset = 0;
    for (offset, chunk) in &chunk_indices {
        assert!(
            *offset >= last_offset,
            "offsets should be non-decreasing even with trimming"
        );
        let token_count = cl100k_base().count(chunk);
        assert!(
            token_count <= 40,
            "chunk exceeded token capacity: {token_count} tokens in `{chunk}`"
        );
        last_offset = *offset;
    }
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_splitter_respects_bpe_token_limits() {
    let tokenizer = o200k_base();
    let splitter = MarkdownSplitter::new(
        ChunkConfig::new(30)
            .with_trim(false)
            .with_sizer(tokenizer),
    );

    let text = r#"# Title

This paragraph should be split on Markdown boundaries but still respect a
token budget. **Bold words** and `inline code` stay intact.

- Bullet one with a few words
- Bullet two with some more words

> A short quote block to exercise block-level splitting.
"#;

    let chunk_indices = splitter.chunk_indices(text).collect::<Vec<_>>();
    assert!(!chunk_indices.is_empty());

    let mut previous_end = 0;
    for (offset, chunk) in &chunk_indices {
        assert_eq!(*offset, previous_end, "chunks should be contiguous");
        previous_end = offset + chunk.len();

        let token_count = o200k_base().count(chunk);
        assert!(
            token_count <= 30,
            "chunk exceeded token capacity: {token_count} tokens in `{chunk}`"
        );
    }

    assert_eq!(previous_end, text.len(), "chunks should cover the full input");

    let reconstructed: String = chunk_indices.iter().map(|(_, chunk)| *chunk).collect();
    assert_eq!(reconstructed, text, "reconstructed text should match input");

    assert!(
        chunk_indices[0].1.starts_with("# Title"),
        "first chunk should start with the heading"
    );
}

#[cfg(feature = "markdown")]
#[test]
fn symbolic_markdown_fixture_splits_cleanly_with_bpe() {
    let tokenizer = cl100k_base();
    let splitter = MarkdownSplitter::new(
        ChunkConfig::new(80)
            .with_trim(false)
            .with_sizer(tokenizer),
    );

    let chunk_indices = splitter
        .chunk_indices(SYMBOLIC_MARKDOWN)
        .collect::<Vec<_>>();

    assert!(
        chunk_indices.len() > 1,
        "fixture should produce multiple chunks with BPE sizing"
    );

    let mut previous_end = 0;
    for (offset, chunk) in &chunk_indices {
        assert_eq!(*offset, previous_end, "chunks should be contiguous");
        previous_end = offset + chunk.len();

        let token_count = cl100k_base().count(chunk);
        assert!(
            token_count <= 80,
            "chunk exceeded token capacity: {token_count} tokens in `{chunk}`"
        );
    }

    assert_eq!(
        previous_end,
        SYMBOLIC_MARKDOWN.len(),
        "chunks should cover the full input"
    );

    let reconstructed: String = chunk_indices.iter().map(|(_, chunk)| *chunk).collect();
    assert_eq!(
        reconstructed, SYMBOLIC_MARKDOWN,
        "reconstructed markdown should match input"
    );
}

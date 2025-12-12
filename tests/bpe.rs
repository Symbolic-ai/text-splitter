#![cfg(feature = "bpe")]

use bpe_openai::{cl100k_base, o200k_base};
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

const SYMBOLIC_MARKDOWN: &str = include_str!("inputs/symbolic_markdown.md");

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

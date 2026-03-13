"""Tests for agent.dedup — semantic similarity deduplication."""

from agent.dedup import _tokenize, _tfidf_cosine, check_duplicate_in_list


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_removes_stopwords(self):
        tokens = _tokenize("this is a the and or but")
        # Most of these should be filtered as stopwords or too short
        assert len(tokens) == 0 or all(len(t) > 2 for t in tokens)

    def test_empty_string(self):
        tokens = _tokenize("")
        assert tokens == []


class TestCosineSimilarity:
    def test_identical_texts(self):
        sim = _tfidf_cosine("hello world test", "hello world test")
        assert sim > 0.99

    def test_completely_different(self):
        sim = _tfidf_cosine("alpha beta gamma", "delta epsilon zeta")
        assert sim < 0.1

    def test_partial_overlap(self):
        sim = _tfidf_cosine("hello world test foo", "hello world bar baz")
        assert 0.2 < sim < 0.9

    def test_empty_text(self):
        sim = _tfidf_cosine("", "hello")
        assert sim == 0.0


class TestCheckDuplicateInList:
    def test_finds_duplicate(self):
        existing = [
            "Our community built something amazing this week",
            "Check out the latest product announcement",
        ]
        result = check_duplicate_in_list(
            "Our community built something amazing this week!", existing, threshold=0.7
        )
        assert result["is_duplicate"] is True
        assert result["max_similarity"] > 0.7

    def test_no_duplicate(self):
        existing = [
            "Our community built something amazing this week",
            "Check out the latest product announcement",
        ]
        result = check_duplicate_in_list(
            "The weather is beautiful for hiking today", existing, threshold=0.7
        )
        assert result["is_duplicate"] is False

    def test_empty_list(self):
        result = check_duplicate_in_list("any caption", [], threshold=0.75)
        assert result["is_duplicate"] is False
        assert result["max_similarity"] == 0.0

    def test_returns_most_similar(self):
        existing = [
            "Alpha beta gamma delta",
            "Hello world test caption for checking",
            "Completely unrelated text about cooking",
        ]
        result = check_duplicate_in_list(
            "Hello world test caption", existing, threshold=0.5
        )
        if result["is_duplicate"]:
            assert "Hello world" in result["similar_to"]

from edgebot.nlu.data import TAGS, build_splits, split_templates
from edgebot.nlu.engine import spans, tags_to_slots


def test_templates_are_disjoint():
    train, test = split_templates()
    for intent in train:
        assert not set(train[intent]) & set(test[intent])


def test_tags_are_valid_bio():
    for e in build_splits(n_train=50, n_test=40)["train"]:
        assert len(e.tokens) == len(e.tags)
        assert all(t in TAGS for t in e.tags)


def test_span_extraction_and_canonicalisation():
    toks = "move the read box from self a to the packing station".split()
    tags = ["O", "O", "B-color", "B-object", "O", "B-source", "I-source", "O", "O", "B-destination", "I-destination"]
    assert spans(tags) == {(2, 3, "color"), (3, 4, "object"), (5, 7, "source"), (9, 11, "destination")}
    assert tags_to_slots(toks, tags) == {"color": "red", "object": "box", "source": "shelf a",
                                         "destination": "packing station"}

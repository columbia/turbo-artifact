from loguru import logger

from privacypacking.budget import Budget


def test_gt():
    b1 = Budget({1: 1, 2: 2})
    b2 = Budget({1: 0.5, 2: 3})
    b3 = Budget({1: 2, 2: 3})

    assert b1 >= b2

    assert b2 >= b1

    logger.info(b1 - b3)
    logger.info(b1 >= b3)

    assert not (b1 >= b3)


def test_immutable():
    o1 = {1: 1, 2: 2}
    b = Budget(o1)

    original_alphas = b.alphas

    o1[3] = 4

    assert original_alphas == b.alphas

    assert len(b.alphas) == 2


def test_same_support():
    b1 = Budget({1: 1, 2: 2})
    b2 = Budget({1: 0.5, 2: 3, 4: 10})

    assert b1.alphas != b2.alphas

    c1, c2 = Budget.same_support(b1, b2)

    assert c1.alphas == c2.alphas

    s = b1 + b2

    assert s.alphas == b1.alphas
    assert s.alphas != b2.alphas

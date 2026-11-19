import sqlite3

import pytest

from semantic_cache import HashEmbedder


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "mini.db")
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE orders (order_id TEXT, order_status TEXT, purchased TEXT);
        INSERT INTO orders VALUES
            ('a', 'delivered', '2017-01-05'),
            ('b', 'delivered', '2017-02-10'),
            ('c', 'canceled', '2017-02-11'),
            ('d', 'delivered', '2017-03-01');
        """
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def embedder():
    return HashEmbedder()

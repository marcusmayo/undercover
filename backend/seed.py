"""Rebuild the UnderCover database from data/."""

from backend import db

if __name__ == "__main__":
    db.seed()
    print(f"Seeded {db.engine().url.render_as_string(hide_password=True)}")

"""Quick verification of the ingested reference database (read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from config import CONFIG  # noqa: E402
from db.database import Database  # noqa: E402
from db.models import Drug, Ingredient, Interaction  # noqa: E402

database = Database(CONFIG.database)
with database.session() as session:
    n_ing = session.scalar(select(func.count(Ingredient.id)))
    n_drugs = session.scalar(select(func.count(Drug.id)))
    n_int = session.scalar(select(func.count(Interaction.id)))
    severities = session.execute(
        select(Interaction.severity, func.count(Interaction.id)).group_by(Interaction.severity)
    ).all()
    sample = session.scalars(
        select(Interaction).where(Interaction.severity == "Major").limit(3)
    ).all()

print(f"ingredients={n_ing}  drugs={n_drugs}  interactions={n_int}")
print("severity counts:", dict(severities))
for row in sample:
    a = session.get(Ingredient, row.ingredient_a_id).name
    b = session.get(Ingredient, row.ingredient_b_id).name
    print(f"  Major: {a} + {b} (source={row.source})")

"""Add user datasets schema.

Revision ID: 20261005_datasets
Revises: 20260919_initial
Create Date: 2026-10-05
"""
from alembic import op
from app.db.session import Base
import app.db.models  # noqa: F401

# revision identifiers, used by Alembic.
revision = "20261005_datasets"
down_revision = "20260919_initial"
branch_labels = None
depends_on = None

def upgrade():
    Base.metadata.create_all(bind=op.get_bind())

def downgrade():
    bind = op.get_bind()
    if "datasets" in Base.metadata.tables:
        Base.metadata.tables["datasets"].drop(bind=bind, checkfirst=True)

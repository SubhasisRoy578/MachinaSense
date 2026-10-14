"""Add model_registry schema.

Revision ID: 20261008_model_registry
Revises: 20261005_training_jobs
Create Date: 2026-10-08
"""
from alembic import op
from app.db.session import Base
import app.db.models  # noqa: F401

# revision identifiers, used by Alembic.
revision = "20261008_model_registry"
down_revision = "20261005_training_jobs"
branch_labels = None
depends_on = None

def upgrade():
    Base.metadata.create_all(bind=op.get_bind())

def downgrade():
    bind = op.get_bind()
    if "model_registry" in Base.metadata.tables:
        Base.metadata.tables["model_registry"].drop(bind=bind, checkfirst=True)

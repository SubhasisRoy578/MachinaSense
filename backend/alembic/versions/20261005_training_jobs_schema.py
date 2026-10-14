"""Add training_jobs schema.

Revision ID: 20261005_training_jobs
Revises: 20261005_datasets
Create Date: 2026-10-05
"""
from alembic import op
from app.db.session import Base
import app.db.models  # noqa: F401

# revision identifiers, used by Alembic.
revision = "20261005_training_jobs"
down_revision = "20261005_datasets"
branch_labels = None
depends_on = None

def upgrade():
    Base.metadata.create_all(bind=op.get_bind())

def downgrade():
    bind = op.get_bind()
    if "training_jobs" in Base.metadata.tables:
        Base.metadata.tables["training_jobs"].drop(bind=bind, checkfirst=True)

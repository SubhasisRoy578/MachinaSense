"""Add model_predictions schema.

Revision ID: 20261008_model_predictions
Revises: 20261008_model_registry
Create Date: 2026-10-08
"""
from alembic import op
from app.db.session import Base
import app.db.models  # noqa: F401

# revision identifiers, used by Alembic.
revision = "20261008_model_predictions"
down_revision = "20261008_model_registry"
branch_labels = None
depends_on = None

def upgrade():
    Base.metadata.create_all(bind=op.get_bind())

def downgrade():
    bind = op.get_bind()
    if "model_predictions" in Base.metadata.tables:
        Base.metadata.tables["model_predictions"].drop(bind=bind, checkfirst=True)

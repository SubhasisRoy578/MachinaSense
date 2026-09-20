"""Initial authenticated MachinaSense schema.

Revision ID: 20260919_initial
"""
from alembic import op
from app.db.session import Base
import app.db.models  # noqa: F401

revision = "20260919_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    Base.metadata.create_all(bind=op.get_bind())

def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())

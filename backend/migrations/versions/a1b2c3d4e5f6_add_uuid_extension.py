"""add uuid extension

Revision ID: a1b2c3d4e5f6
Revises: f8a5e0d49c2d
Create Date: 2025-09-14 10:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f8a5e0d49c2d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

def downgrade() -> None:
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')

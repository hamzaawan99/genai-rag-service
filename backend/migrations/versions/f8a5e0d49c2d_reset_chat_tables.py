"""reset chat tables

Revision ID: f8a5e0d49c2d
Revises: 1a2b3c4d5e6f
Create Date: 2025-09-14 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f8a5e0d49c2d'
down_revision: Union[str, None] = '1a2b3c4d5e6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Drop everything related to chat tables
    op.execute("""
        DO $$ 
        BEGIN
            -- Drop indexes if they exist
            DROP INDEX IF EXISTS ix_chat_messages_id;
            DROP INDEX IF EXISTS ix_chat_messages_session_id;
            DROP INDEX IF EXISTS ix_chat_messages_created_at;
            DROP INDEX IF EXISTS ix_chat_sessions_id;
            DROP INDEX IF EXISTS ix_chat_sessions_session_id;
            DROP INDEX IF EXISTS ix_chat_sessions_knowledge_base_id;
            DROP INDEX IF EXISTS ix_chat_sessions_user_id;
            DROP INDEX IF EXISTS ix_chat_sessions_created_at;
            
            -- Drop tables if they exist
            DROP TABLE IF EXISTS chat_messages CASCADE;
            DROP TABLE IF EXISTS chat_sessions CASCADE;
            
            -- Drop sequences if they exist
            DROP SEQUENCE IF EXISTS chat_messages_id_seq CASCADE;
            DROP SEQUENCE IF EXISTS chat_sessions_id_seq CASCADE;
        END $$;
    """)

def downgrade() -> None:
    pass # No downgrade since we're just cleaning up

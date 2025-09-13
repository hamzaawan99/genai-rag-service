"""Update knowledge base table

Revision ID: 1a2b3c4d5e6f
Revises: 
Create Date: 2025-09-08 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '1a2b3c4d5e6f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns
    op.add_column('knowledge_bases', sa.Column('vector_db', sa.String(), nullable=False, server_default='chromadb'))
    op.add_column('knowledge_bases', sa.Column('collection_name', sa.String(), nullable=False))
    
    # Drop old column
    op.drop_column('knowledge_bases', 'weaviate_class_name')
    
    # Create unique constraint on collection_name
    op.create_unique_constraint('uq_knowledge_bases_collection_name', 'knowledge_bases', ['collection_name'])
    
    # Update document_metadata column type to JSONB
    op.alter_column('documents', 'document_metadata',
                    type_=postgresql.JSONB,
                    postgresql_using='document_metadata::jsonb')


def downgrade() -> None:
    # Remove unique constraint
    op.drop_constraint('uq_knowledge_bases_collection_name', 'knowledge_bases')
    
    # Add back old column
    op.add_column('knowledge_bases', sa.Column('weaviate_class_name', sa.String(), nullable=False))
    
    # Drop new columns
    op.drop_column('knowledge_bases', 'vector_db')
    op.drop_column('knowledge_bases', 'collection_name')
    
    # Revert document_metadata column type to TEXT
    op.alter_column('documents', 'document_metadata',
                    type_=sa.Text,
                    postgresql_using='document_metadata::text')

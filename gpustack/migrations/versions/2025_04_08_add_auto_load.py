"""add auto_load column to models table

Revision ID: 9e2571d3a211
Revises: c45e397531d1
Create Date: 2025-04-08 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from typing import Union, Sequence

# revision identifiers, used by Alembic.
revision = '9e2571d3a211'
down_revision = 'c45e397531d1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add auto_load column to models table with default value 1 (enabled)
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('auto_load', sa.Integer(), server_default='1', nullable=False))


def downgrade() -> None:
    # Remove auto_load column from models table
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('auto_load') 

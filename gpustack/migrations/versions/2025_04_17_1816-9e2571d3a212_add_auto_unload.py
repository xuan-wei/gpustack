"""add auto_unload columns to models table

Revision ID: 9e2571d3a212
Revises: 9e2571d3a211
Create Date: 2025-05-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from typing import Union, Sequence

# revision identifiers, used by Alembic.
revision = '9e2571d3a212'
down_revision = '9e2571d3a211'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add auto_unload related columns to models table
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('auto_unload', sa.Integer(), server_default='0', nullable=False))
        batch_op.add_column(sa.Column('auto_unload_timeout', sa.Integer(), server_default='120', nullable=False))
        batch_op.add_column(sa.Column('last_request_time', sa.DateTime(), nullable=True))
    
def downgrade() -> None:
    # Remove auto_unload related columns from models table
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('auto_unload')
        batch_op.drop_column('auto_unload_timeout')
        batch_op.drop_column('last_request_time') 

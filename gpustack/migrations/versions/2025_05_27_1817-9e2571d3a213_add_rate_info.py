"""add request tracking metrics columns to models table

Revision ID: 9e2571d3a213
Revises: 9e2571d3a212
Create Date: 2025-04-17 18:17:00.000000

"""
from alembic import op
import sqlalchemy as sa
from typing import Union, Sequence

# revision identifiers, used by Alembic.
revision = '9e2571d3a213'
down_revision = '9e2571d3a212'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add request tracking metrics columns to models table
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('auto_adjust_replicas', sa.Integer(), server_default='0', nullable=False))
        batch_op.add_column(sa.Column('avg_request_rate', sa.Float(), server_default='0.0', nullable=False))
        batch_op.add_column(sa.Column('avg_process_rate', sa.Float(), server_default='0.0', nullable=False))
        batch_op.add_column(sa.Column('last_scale_time', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('last_scale_message', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove request tracking metrics columns from models table
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('auto_adjust_replicas')
        batch_op.drop_column('avg_request_rate')
        batch_op.drop_column('avg_process_rate')
        batch_op.drop_column('last_scale_time')
        batch_op.drop_column('last_scale_message')

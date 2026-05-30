"""add custom v2.1.2 fields (model lifecycle, scaling, gpu memory range, instance draining)

Revision ID: 2026_04_14_0001
Revises: 8ad0f94c92e8
Create Date: 2026-04-14 00:01:00.000000

Consolidated single migration that introduces every schema change required by
our v2.1.2 custom fork:

- `models`: auto_load / auto_unload lifecycle flags + thresholds + GPU memory
  range hints used by the vLLM placement scorer
- `model_instances`: `is_draining` flag used by AutoDrainTask to eliminate the
  scale-down race against Higress endpoint convergence

Down-revision points directly at upstream v2.1.2 head (8ad0f94c92e8) so the
chain has a single attachment point during future rebases.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


# revision identifiers, used by Alembic.
revision: str = '2026_04_14_0001'
down_revision: Union[str, None] = '8ad0f94c92e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


MODEL_COLUMNS_TO_ADD = {
    # ── Auto-load / auto-unload lifecycle ────────────────────────────────────
    'auto_load': sa.Column('auto_load', sa.Integer(), nullable=False, server_default='1'),
    'auto_load_replicas': sa.Column('auto_load_replicas', sa.Integer(), nullable=False, server_default='1'),
    'auto_unload': sa.Column('auto_unload', sa.Integer(), nullable=False, server_default='0'),
    'auto_unload_timeout': sa.Column('auto_unload_timeout', sa.Integer(), nullable=False, server_default='10'),
    'auto_adjust_replicas': sa.Column('auto_adjust_replicas', sa.Integer(), nullable=False, server_default='0'),
    # ── Rolling metrics + last-event timestamps used by lifecycle tasks ──────
    'avg_request_rate': sa.Column('avg_request_rate', sa.Float(), nullable=False, server_default='0'),
    'avg_process_rate': sa.Column('avg_process_rate', sa.Float(), nullable=False, server_default='0'),
    'last_request_time': sa.Column('last_request_time', UTCDateTime(), nullable=True),
    'last_scale_time': sa.Column('last_scale_time', UTCDateTime(), nullable=True),
    'last_scale_message': sa.Column('last_scale_message', sa.Text(), nullable=True),
    # ── Auto-scaling thresholds (consumed by AutoScalingTask) ───────────────
    'scale_window_minutes': sa.Column('scale_window_minutes', sa.Integer(), nullable=False, server_default='5'),
    'scale_down_kv_threshold': sa.Column('scale_down_kv_threshold', sa.Float(), nullable=False, server_default='0.6'),
    # ── GPU memory occupancy hints (read by vllm placement policies) ─────────
    'gpu_memory_min_gib': sa.Column('gpu_memory_min_gib', sa.Integer(), nullable=True),
    'gpu_memory_max_gib': sa.Column('gpu_memory_max_gib', sa.Integer(), nullable=True),
}


MODEL_INSTANCE_COLUMNS_TO_ADD = {
    # Drained instances stay in the DB until AutoDrainTask confirms Higress has
    # converged on the new endpoint set, then deletes them.
    'is_draining': sa.Column(
        'is_draining',
        sa.Boolean(),
        nullable=False,
        server_default=sa.false(),
    ),
}


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        for column in MODEL_COLUMNS_TO_ADD.values():
            batch_op.add_column(column)

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        for column in MODEL_INSTANCE_COLUMNS_TO_ADD.values():
            batch_op.add_column(column)


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        for column_name in reversed(list(MODEL_INSTANCE_COLUMNS_TO_ADD.keys())):
            batch_op.drop_column(column_name)

    with op.batch_alter_table('models', schema=None) as batch_op:
        for column_name in reversed(list(MODEL_COLUMNS_TO_ADD.keys())):
            batch_op.drop_column(column_name)

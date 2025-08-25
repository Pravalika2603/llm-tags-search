#!/usr/bin/env bash
set -e
psql "$DATABASE_URL" -f alembic.sql

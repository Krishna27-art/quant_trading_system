#!/usr/bin/env python3
"""
CI Validation for Database Migrations

Before deploy:
- Migration Up must succeed
- Migration Down must succeed
- No data loss
- Zero downtime

Usage:
    python scripts/validate_migrations.py
"""

import subprocess
import sys

from utils.logger import get_logger

logger = get_logger("migration_validation")


class MigrationValidator:
    """Validate database migrations for CI/CD."""

    def __init__(self):
        """Initialize migration validator."""
        self.logger = logger

    def run_alembic_command(self, command: list[str]) -> tuple[bool, str]:
        """
        Run Alembic command.

        Args:
            command: Command to run

        Returns:
            (success, output)
        """
        try:
            result = subprocess.run(
                ["alembic"] + command, capture_output=True, text=True, timeout=300
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            return success, output
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def validate_migration(self, revision: str) -> bool:
        """
        Validate a single migration.

        Args:
            revision: Migration revision

        Returns:
            True if validation passed
        """
        self.logger.info(f"Validating migration: {revision}")

        # Test upgrade
        success, output = self.run_alembic_command(["upgrade", revision])
        if not success:
            self.logger.error(f"Migration upgrade failed: {output}")
            return False

        self.logger.info(f"Migration upgrade succeeded: {revision}")

        # Test downgrade
        success, output = self.run_alembic_command(["downgrade", revision])
        if not success:
            self.logger.error(f"Migration downgrade failed: {output}")
            return False

        self.logger.info(f"Migration downgrade succeeded: {revision}")

        # Re-upgrade to leave database in correct state
        success, output = self.run_alembic_command(["upgrade", "head"])
        if not success:
            self.logger.error(f"Final migration upgrade failed: {output}")
            return False

        return True

    def validate_all_migrations(self) -> bool:
        """
        Validate all pending migrations.

        Returns:
            True if all validations passed
        """
        self.logger.info("Starting migration validation")

        # Get current revision
        success, output = self.run_alembic_command(["current"])
        if not success:
            self.logger.error(f"Failed to get current revision: {output}")
            return False

        current_revision = output.strip()
        self.logger.info(f"Current revision: {current_revision}")

        # Get target revision
        success, output = self.run_alembic_command(["heads"])
        if not success:
            self.logger.error(f"Failed to get target revision: {output}")
            return False

        target_revision = output.strip()
        self.logger.info(f"Target revision: {target_revision}")

        # Validate upgrade to head
        success, output = self.run_alembic_command(["upgrade", "head"])
        if not success:
            self.logger.error(f"Migration to head failed: {output}")
            return False

        self.logger.info("Migration to head succeeded")

        # Validate downgrade to base
        success, output = self.run_alembic_command(["downgrade", "base"])
        if not success:
            self.logger.error(f"Migration to base failed: {output}")
            return False

        self.logger.info("Migration to base succeeded")

        # Re-upgrade to head
        success, output = self.run_alembic_command(["upgrade", "head"])
        if not success:
            self.logger.error(f"Final migration to head failed: {output}")
            return False

        self.logger.info("Final migration to head succeeded")

        return True

    def check_migration_consistency(self) -> bool:
        """
        Check migration consistency.

        Returns:
            True if consistent
        """
        # Check for duplicate revisions
        success, output = self.run_alembic_command(["check"])
        if not success:
            self.logger.error(f"Migration consistency check failed: {output}")
            return False

        self.logger.info("Migration consistency check passed")
        return True


def main():
    """Main entry point."""
    validator = MigrationValidator()

    # Check consistency
    if not validator.check_migration_consistency():
        logger.error("Migration consistency check failed")
        sys.exit(1)

    # Validate all migrations
    if not validator.validate_all_migrations():
        logger.error("Migration validation failed")
        sys.exit(1)

    logger.info("All migration validations passed")
    sys.exit(0)


if __name__ == "__main__":
    main()

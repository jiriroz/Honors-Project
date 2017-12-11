#!/usr/bin/env python
import os
import sys

from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = "Runs an arbitrary module, in the Django environment, for quick prototyping of code that's too big for the shell."

    def handle(self, *args, **options):
        if not args:
            return
        module_name = args[0]

        try:
            __import__(module_name)
        except ImportError:
            print("Unable to import module %s.  Check that is within Django's PYTHONPATH" % (module_name))



if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[1] == "runmodule":
            cmd = Command()
            cmd.handle(*(sys.argv[2:]))
            sys.exit(0)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "honors.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

"""Project-local test package.

Keeping this directory importable prevents an unrelated third-party ``tests``
package in a runtime image from shadowing shared G0 test helpers.
"""

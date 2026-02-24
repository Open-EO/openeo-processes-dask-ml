import pytest

from openeo_processes_dask_ml.process_implementations.utils.model_cache_utils import (
    url_to_dir_string,
)


@pytest.mark.parametrize(
    "input_url, expected_output",
    [
        (
            "https://www.example.com/some/path?query=string&id=123#fragment",
            "https___www_example_com_some_path_query_string_id_123_fragment",
        ),
        ("http://example.com/a:b/c*d?e", "http___example_com_a_b_c_d_e"),
        ("CON", "_CON"),  # Windows reserved name
        ("PRN.", "_PRN"),  # Windows reserved name with trailing dot
        ("AUX ", "_AUX"),  # Windows reserved name with trailing space (stripped)
        (
            "http://example.com/file.name.",
            "http___example_com_file_name",
        ),  # Trailing dot
        (
            "http://example.com/file name with spaces",
            "http___example_com_file_name_with_spaces",
        ),  # Spaces
        (
            "http://example.com/with|pipe",
            "http___example_com_with_pipe",
        ),  # Pipe character
        (
            'http://example.com/with"quote',
            "http___example_com_with_quote",
        ),  # Quote character
        (
            "http://example.com/with<angle>brackets",
            "http___example_com_with_angle_brackets",
        ),  # Angle brackets
        ("just_a_valid_name", "just_a_valid_name"),  # Already valid name
        ("", "model"),  # Empty string
        (
            '?*<>|"',
            "sanitized_dir",
        ),  # String with only problematic characters (should become _)
        (" .", "sanitized_dir"),  # String that becomes just '.' or ' ' after stripping
    ],
)
def test_sanitize_url_for_directory(input_url, expected_output):
    """
    Tests the sanitize_url_for_directory function with various inputs.
    """
    actual_output = url_to_dir_string(input_url)
    assert actual_output == expected_output

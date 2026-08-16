import json
import unittest
from pathlib import Path


class WorkbookExecutionTests(unittest.TestCase):
    def test_code_cells_execute_top_to_bottom(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebook_path = root / "notebooks" / "thesis_evidence_workbook.ipynb"
        notebook = json.loads(notebook_path.read_text())
        namespace = {"__name__": "__canonical_workbook_test__"}
        previous_cwd = Path.cwd()
        try:
            # Validate the same two supported launch locations: repository root
            # and the notebooks directory.
            import os

            os.chdir(root)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)
        finally:
            os.chdir(previous_cwd)

        result = namespace["result"]
        self.assertEqual(result.comparisons, 173)
        self.assertEqual(result.operational_alerts, 1)
        self.assertEqual(result.cost_common_n, 1113)


if __name__ == "__main__":
    unittest.main()

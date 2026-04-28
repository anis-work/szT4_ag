"""CV Ingestion Plugin for Semantic Kernel.

Exposes file-based CV loading as a kernel_function so it can be
invoked through the kernel like any other SK plugin.
"""

from typing import Annotated, List
from semantic_kernel.functions import kernel_function

from pdf_loader import load_cvs_from_folder
from models import CV


class CVIngestionPlugin:
    """SK plugin that loads CVs from a folder of PDF/DOCX files."""

    @kernel_function(description="Load CVs from a folder of PDF or DOCX files")
    def load_from_folder(
        self,
        folder: Annotated[str, "Path to folder containing PDF/DOCX resume files"],
    ) -> str:
        """Load all resumes from folder and return a summary string.

        The actual CV objects are stored on the instance for the pipeline
        to retrieve via .cvs after invocation.
        """
        self.cvs: List[CV] = load_cvs_from_folder(folder)
        names = [cv.candidate_name for cv in self.cvs]
        return f"Loaded {len(self.cvs)} CVs: {', '.join(names)}"

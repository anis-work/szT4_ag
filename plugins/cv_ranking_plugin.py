"""CV Ranking Plugin for Semantic Kernel."""

import json
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel import Kernel


class CVRankingPlugin:
    def __init__(self, kernel: Kernel) -> None:
        self._kernel = kernel

    @kernel_function(description="Rank candidates based on job description")
    async def rank(
        self,
        job_description: Annotated[str, "The job description and requirements"],
        retrieved_cvs: Annotated[str, "Formatted string of retrieved candidate CVs"],
    ) -> str:
        rank_fn = self._kernel.get_function(plugin_name="ranking", function_name="rank_candidates")
        result = await self._kernel.invoke(
            rank_fn,
            job_description=job_description,
            retrieved_cvs=retrieved_cvs,
        )
        result_text = str(result).strip()
        json.loads(result_text)  # validate JSON
        return result_text

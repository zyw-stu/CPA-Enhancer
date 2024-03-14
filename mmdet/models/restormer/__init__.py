from .prompt_restormer import PromptRestormer

custom_imports = dict(
    imports=['mmdet.models.restormer.prompt_restormer'],
    allow_failed_imports=False)

__all__ = ['PromptRestormer']
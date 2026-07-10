import os
import json
import types
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any
import io
import sys
import doctest
from doctest import DocTestParser
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import warnings
warnings.filterwarnings('ignore')

def load_and_flatten_all_jsons(input_dir: str) -> pd.DataFrame:
    all_rows = []
    
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path) as f:
            data = json.load(f)

        for entry in data:
            question = entry["question_name"]
            for block in entry["code_block"]:
                for source_type in ["gt", "synthetic"]:
                    all_rows.append({
                        "question": question,
                        "quantile": block.get("quantile", "submission_q0"),
                        "source": source_type,
                        "code": block[source_type]
                    })

    return pd.DataFrame(all_rows)


class Autograder:
    
    def __init__(self, 
                 submissions: pd.DataFrame, 
                 test_files_dir: str,
                 code_col_name: str,
                 graded_submissions: Optional[pd.DataFrame] = None):
        """
        Initialize the autograder with global variables, submissions, and test files directory.
        Args:
            submissions: DataFrame containing student code submissions
            test_files_dir: Directory containing test files for each assignment
            code_col_name: Column name containing the code submissions
            graded_submissions: Optional DataFrame to store graded results
        """
        self.submissions = submissions
        self.test_files_dir = test_files_dir
        self.graded_submissions = graded_submissions
        self.code_col_name = code_col_name
        self._configure_multiprocessing()

    @staticmethod
    def _configure_multiprocessing() -> None:
        """Configure multiprocessing settings for proper execution."""
        multiprocessing.set_start_method('fork', force=True)
            
    @staticmethod
    def _redirect_output() -> Tuple[io.StringIO, Any, Any]:
        """
        Set up output redirection for capturing test results.
        
        Returns:
            Tuple containing output buffer and original stdout/displayhook
        """
        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        original_displayhook = sys.displayhook
        return output_buffer, original_stdout, original_displayhook
        
    @staticmethod
    def _create_display_hook(output_buffer: io.StringIO, test_globals: Dict[str, Any]):
        """
        Create a custom display hook for handling test output.
        
        Args:
            output_buffer: Buffer to capture output
            test_globals: Global variables dictionary for the test
        """
        def display_hook(value: Any) -> None:
            if value is not None:
                output_buffer.write(repr(value) + "\n")
                test_globals['_'] = value
        return display_hook

    @classmethod
    def execute_single_test(cls,
                          example: doctest.Example,
                          test_globals: Dict[str, Any],
                          compile_flags: int,
                          option_flags: int) -> Tuple[bool, str, Optional[str]]:
        """
        Execute a single doctest example in an isolated environment.
        
        Args:
            example: Doctest example to execute
            test_globals: Global variables for the test environment
            compile_flags: Flags for code compilation
            option_flags: Doctest option flags
            
        Returns:
            Tuple containing:
            - Boolean indicating if test passed
            - Captured output string
            - Error message (if any)
        """
        
        output_buffer, orig_stdout, orig_displayhook = cls._redirect_output()
        error_message = None
        
        sys.stdout = output_buffer
        sys.displayhook = cls._create_display_hook(output_buffer, test_globals)
        
        try:
            expr = example.source.strip()
            wrapped = f"print({expr})"
            code = compile(wrapped, "<doctest>", "exec")
            exec(code, test_globals)
        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            output_buffer.write(traceback.format_exc())
        finally:
            sys.stdout = orig_stdout
            sys.displayhook = orig_displayhook
            
        result = output_buffer.getvalue()
        # normalize expected
        want = example.want.replace('"""', '').rstrip() + "\n"
        # now compare
        checker = doctest.OutputChecker()
        passed = checker.check_output(want, result, option_flags)
        return passed, result, error_message


    @classmethod
    def execute_test_with_timeout(cls,
                                example: doctest.Example,
                                test_globals: Dict[str, Any],
                                compile_flags: int,
                                option_flags: int,
                                timeout: int = 2) -> Tuple[bool, str, str]:
        """
        Execute a doctest example with timeout protection.
        
        Args:
            example: Doctest example to execute
            test_globals: Global variables for test environment
            compile_flags: Flags for code compilation
            option_flags: Doctest option flags
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple containing test results and any error messages
        """
        queue = multiprocessing.Queue()
        
        def _execute_test_in_process():
            result = cls.execute_single_test(example, test_globals, compile_flags, option_flags)
            queue.put(result)
        
        process = multiprocessing.Process(target=_execute_test_in_process)
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return False, "", "TimeoutError: infinite loop / recursion detected"
            
        return queue.get() if not queue.empty() else (False, "", "Error: No result returned from process")

    @classmethod
    def grade_submission(cls,
                        code_str: str,
                        test_file: str,
                        timeout: int = 2) -> Dict[str, Any]:
        """
        Grade a code submission by running tests from a separate test file.
        
        Args:
            code_str: String containing the code to test
            test_file: Path to the test file containing doctests
            timeout: Maximum execution time per test in seconds
            
        Returns:
            Dictionary containing grading results and test case details
        """
        # Compilation check for submission
        try:
            compiled_code = compile(code_str, "<string>", "exec")
        except Exception as e:
            return {
                "error_type": f"Compilation Error ({type(e).__name__}: {str(e)})",
                "test_cases": None
            }
            
        # Create test module with submission and extra globals
        test_module = types.ModuleType("submission_module")
            
        # Execute submission code
        try:
            exec(compiled_code, test_module.__dict__)
        except Exception as e:
             return {
                 "error_type": f"Execution Error ({type(e).__name__}: {str(e)})",
                 "test_cases": None
             }
        # Load & exec the full test file (helpers + stub), then override stub
        try:
            test_src = open(test_file, 'r').read()
        except Exception as e:
            return {
                "error_type": f"Test File Loading Error ({type(e).__name__}: {str(e)})",
                "test_cases": None
            }

        # bring in helpers (imports, lambdas, fixed code) and the stub
        exec(test_src, test_module.__dict__)
        # now override the stub with the student's submission
        exec(compiled_code, test_module.__dict__)

        # parse only the >>> examples from the original source
        parser = DocTestParser()
        test   = parser.get_doctest(
            test_src,
            globs=test_module.__dict__,
            name=test_file,
            filename=test_file,
            lineno=0
        )


        test_results = []
        overall_error = None

        for example in test.examples:
            passed, output, error = cls.execute_test_with_timeout(
                example,
                test.globs,
                getattr(test, 'compile_flags', 0),
                getattr(test, 'optionflags', 0),
                timeout
            )
            if error:
                overall_error = f"Runtime Error ({error})"
            exp = example.want.replace('"""', '').strip()
            test_results.append({
                    "test_case":     example.source.strip(),
                    "expected":      exp,
                    "got":           output.strip(),
                    "passed":        passed,
                    "error_message": error
                })

        if overall_error is None:
            overall_error = "No Error" if all(t["passed"] for t in test_results) else "Logical Error"

        # compute pass rate
        total = len(test_results)
        passed_count = sum(1 for t in test_results if t["passed"])
        pass_rate = passed_count / total if total else 0.0

        return {
            "error_type":     overall_error,
            "test_pass_rate": pass_rate,
            "test_cases":     test_results
        }



    @staticmethod
    def _process_submission(args: Tuple[int, str, str, int, Dict[str, Any], Dict[str, Any], bool]) -> Tuple[int, Dict[str, Any]]:
        """
        Static method to process a single submission, suitable for multiprocessing.
        
        Args:
            args: Tuple containing:
                - submission_idx: Index of the submission
                - code: Code string to grade
                - test_file: Path to the test file for the question
                - timeout: Maximum execution time
                - shared_memory: Shared memory for caching
                - test_results: Existing test results (if any)
                - rerun: Whether to re-run the tests if there are existing results
                
        Returns:
            Tuple of (submission index, test results)
        """
        submission_idx, code, test_file, timeout, shared_memory, existing_test_results, rerun = args
        
        # Check cache first
        if code.strip() in shared_memory:
            return submission_idx, shared_memory[code.strip()]
        
        # Skip if test results already exist
        if existing_test_results and not rerun:
            return submission_idx, existing_test_results
        
        # Grade the submission using the static method
        test_results = Autograder.grade_submission(
            code,
            test_file,
            timeout=timeout
        )
        
        shared_memory[code.strip()] = test_results
        return submission_idx, test_results

    def grade_submissions(self, timeout: int = 2, rerun: bool = False) -> None:
        """
        Grade all submissions in parallel using a process pool.
        
        Args:
            timeout: Maximum execution time per test in seconds (default: 2)
            rerun: Whether to re-run all submissions (default: False)
        
        Raises:
            RuntimeError: If there's an error during the grading process
        """
        if self.graded_submissions is None:
            self.graded_submissions = self.submissions.copy()
            self.graded_submissions["test_results"] = None            
            
        try:
            # shared memory for caching
            manager = multiprocessing.Manager()
            shared_memory = manager.dict()
            
            submission_data = [
                (idx, 
                 row[self.code_col_name], 
                 os.path.join(self.test_files_dir, f"{row['question']}.py"),
                 timeout,
                 shared_memory,
                 row["test_results"],
                 rerun)
                for idx, row in self.graded_submissions.iterrows()
            ]
            
            # parallel processing
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self._process_submission, submission_data),
                    total=len(self.submissions),
                    desc="Grading submissions"
                ))
            
            indices, test_results = zip(*results)
            self.graded_submissions.loc[list(indices), "test_results"] = list(test_results)
            
        except Exception as e:
            raise RuntimeError(f"Error during bulk grading: {str(e)}") from e

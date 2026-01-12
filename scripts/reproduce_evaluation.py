
import sys
import os
import shutil
import json
from datetime import datetime

def run_compiled_tests():
    \"\"\"Run all pre-generated ChatGPT tests and collect metrics.
    
    This script:
    1. Iterates through all projects listed in files.txt
    2. Copies each pre-generated test from gpt-tests/ to src/test/
    3. Runs Maven with Pitest for mutation testing
    4. Collects pass/fail results
    5. Saves summary to JSON file
    
    The process can take several hours for all 33 projects.
    \"\"\"
    if not os.path.exists("files.txt"):
        print("Error: files.txt not found in current directory.")
        sys.exit(1)

    with open('files.txt', 'r') as f:
        projects = f.readlines()

    # Results tracking
    results = {}
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for line in projects:
        line = line.strip()
        if not line: continue
        
        parts = line.split(':')
        project_name = parts[0]
        full_class_name = parts[1]
        class_name = full_class_name.replace("ds.", "")
        
        project_dir = os.path.join("..", "projetos", project_name)
        test_dir = os.path.join(project_dir, "src", "test", "java", "ds")
        gpt_tests_dir = os.path.join(project_dir, "gpt-tests")
        
        print(f"\n{'='*60}")
        print(f"Processing project: {project_name}")
        print(f"{'='*60}")
        
        results[project_name] = {"passed": [], "failed": [], "skipped": []}
        
        # Backup existing tests
        backup_dir = os.path.join(project_dir, "src", "test", "java", "ds_backup")
        if os.path.exists(test_dir):
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(test_dir, backup_dir)
            # Clean test dir
            for item in os.listdir(test_dir):
                target_path = os.path.join(test_dir, item)
                if os.path.isfile(target_path) or os.path.islink(target_path):
                    os.remove(target_path)
                elif os.path.isdir(target_path):
                    shutil.rmtree(target_path)
        else:
            os.makedirs(test_dir)

        try:
            for i in range(34):
                test_file_name = f"{class_name}Test{i}.java"
                src_test_file = os.path.join(gpt_tests_dir, test_file_name)
                dst_test_file = os.path.join(test_dir, test_file_name)
                
                if not os.path.exists(src_test_file):
                    print(f"  [SKIP] Test {i}: file not found")
                    results[project_name]["skipped"].append(i)
                    total_skipped += 1
                    continue
                
                shutil.copy(src_test_file, dst_test_file)
                
                print(f"  Running test {i}...", end=" ")
                
                # Construct Maven command (suppress output for cleaner logs)
                cmd = f"cd {project_dir} && mvn -q -DclassName=\"ds.{class_name}\" -DtestName=\"ds.{class_name}Test{i}\" clean install org.pitest:pitest-maven:mutationCoverage > /dev/null 2>&1"
                
                # Execute command
                ret_code = os.system(cmd)
                
                if ret_code == 0:
                    print("[PASS]")
                    results[project_name]["passed"].append(i)
                    total_passed += 1
                else:
                    print("[FAIL]")
                    results[project_name]["failed"].append(i)
                    total_failed += 1
                
                # Clean up this test file
                if os.path.exists(dst_test_file):
                    os.remove(dst_test_file)
                
        finally:
            # Restore backup
            print("  Restoring original tests...")
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            if os.path.exists(backup_dir):
                shutil.move(backup_dir, test_dir)
    
    # Generate summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total Passed:  {total_passed}")
    print(f"Total Failed:  {total_failed}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Tests:   {total_passed + total_failed + total_skipped}")
    print("")
    
    for proj, res in results.items():
        passed_count = len(res["passed"])
        failed_count = len(res["failed"])
        skipped_count = len(res["skipped"])
        print(f"{proj}: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
    
    # Save results to JSON file
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped
            },
            "projects": results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    run_compiled_tests()

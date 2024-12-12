import importlib
import pkg_resources
import sys

def check_dependencies():
    """Check if all required dependencies are installed correctly."""
    
    # Read requirements file
    with open('requirements.txt', 'r') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]
    
    # Check each requirement
    missing = []
    version_mismatch = []
    
    for requirement in requirements:
        # Parse requirement
        name = requirement.split('>=')[0]
        required_version = requirement.split('>=')[1] if '>=' in requirement else None
        
        try:
            # Try to import the module
            module = importlib.import_module(name.replace('-', '_'))
            
            # Check version if specified
            if required_version:
                installed_version = pkg_resources.get_distribution(name).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
                    version_mismatch.append(f"{name} (required: >={required_version}, installed: {installed_version})")
                    
        except ImportError:
            missing.append(name)
    
    # Print results
    if not (missing or version_mismatch):
        print("✅ All dependencies are installed correctly!")
        return True
    
    if missing:
        print("\n❌ Missing dependencies:")
        for package in missing:
            print(f"  - {package}")
            
    if version_mismatch:
        print("\n⚠️ Version mismatches:")
        for package in version_mismatch:
            print(f"  - {package}")
            
    print("\nTo install missing dependencies, run:")
    print("pip install -r requirements.txt")
    
    return False

if __name__ == "__main__":
    check_dependencies() 
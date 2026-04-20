import ctypes
import json
import os

# 1. Load the shared library
lib_path = os.path.abspath("cprf.so")
cprf_lib = ctypes.CDLL(lib_path)

# 2. Define the argument and return types for each C function
# C strings translate to ctypes.c_char_p
cprf_lib.C_KeyGen.argtypes = [ctypes.c_char_p, ctypes.c_int]
cprf_lib.C_KeyGen.restype = ctypes.c_char_p

cprf_lib.C_Constrain.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
cprf_lib.C_Constrain.restype = ctypes.c_char_p

cprf_lib.C_Eval.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
cprf_lib.C_Eval.restype = ctypes.c_char_p

cprf_lib.C_CEval.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
cprf_lib.C_CEval.restype = ctypes.c_char_p

def main():
    # --- Example 1: KeyGen ---
    # Python strings must be encoded to bytes before passing to c_char_p
    modulus_hex = "FFFF" # Example modest modulus
    length = 5
    
    print("--- 1. Generating Master Key ---")
    msk_bytes = cprf_lib.C_KeyGen(modulus_hex.encode('utf-8'), length)
    msk_json = msk_bytes.decode('utf-8')
    print(json.dumps(json.loads(msk_json), indent=2))
    
    # --- Example 2: Eval ---
    print("\n--- 2. Evaluating Master Key ---")
    # Our input x is an array of big integers, represented as a JSON array of hex strings
    x_array = ["A", "B", "C", "D", "E"]
    x_json = json.dumps(x_array)
    
    eval_bytes = cprf_lib.C_Eval(msk_bytes, x_json.encode('utf-8'))
    print("Eval Hash Result (Hex):", eval_bytes.decode('utf-8'))
    
    # --- Example 3: Constrain ---
    print("\n--- 3. Constraining Key ---")
    # For constrain, z must be an array of length 5
    z_array = ["1", "2", "3", "4", "5"]
    z_json = json.dumps(z_array)
    
    csk_bytes = cprf_lib.C_Constrain(msk_bytes, z_json.encode('utf-8'))
    csk_json = csk_bytes.decode('utf-8')
    print(json.dumps(json.loads(csk_json), indent=2))

    # --- Example 4: CEval ---
    print("\n--- 4. Evaluating Constrained Key ---")
    ceval_bytes = cprf_lib.C_CEval(csk_bytes, x_json.encode('utf-8'))
    print("CEval Hash Result (Hex):", ceval_bytes.decode('utf-8'))

if __name__ == "__main__":
    main()

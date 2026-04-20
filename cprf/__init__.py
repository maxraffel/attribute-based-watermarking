import ctypes
import json
import os
from typing import List

# Locate the c-shared library relative to this file
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cprf.so")

if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Shared library not found at: {_lib_path}. Please make sure 'go build -o cprf.so -buildmode=c-shared cprf.go' has been run.")

_cprf_lib = ctypes.CDLL(_lib_path)

# Configure argument and return types
_cprf_lib.C_KeyGen.argtypes = [ctypes.c_char_p, ctypes.c_int]
_cprf_lib.C_KeyGen.restype = ctypes.c_char_p

_cprf_lib.C_Constrain.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_cprf_lib.C_Constrain.restype = ctypes.c_char_p

_cprf_lib.C_Eval.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_cprf_lib.C_Eval.restype = ctypes.c_char_p

_cprf_lib.C_CEval.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_cprf_lib.C_CEval.restype = ctypes.c_char_p


class ConstrainedKey:
    """Represents a CPRF Constrained Key."""
    
    def __init__(self, json_str: str):
        self._json_str = json_str
        self._data = json.loads(json_str)
        self.length = self._data['length']
        self.modulus = int(self._data['modulus'], 16)
        self.z1 = [int(v, 16) for v in self._data['z1']]

    def _to_json_bytes(self) -> bytes:
        return self._json_str.encode('utf-8')

    def c_eval(self, x: List[int]) -> bytes:
        """
        Evaluates the CPRF function with the constrained key.
        :param x: List of large integers.
        :return: A bytes object representing the hashed evaluation result.
        """
        x_hex = [hex(v % self.modulus)[2:] for v in x]
        x_bytes = json.dumps(x_hex).encode('utf-8')
        res_bytes = _cprf_lib.C_CEval(self._to_json_bytes(), x_bytes)
        return bytes.fromhex(res_bytes.decode('utf-8'))


class MasterKey:
    """Represents a CPRF Master Key."""
    
    def __init__(self, json_str: str):
        self._json_str = json_str
        self._data = json.loads(json_str)
        self.length = self._data['length']
        self.modulus = int(self._data['modulus'], 16)
        self.z0 = [int(v, 16) for v in self._data['z0']]

    def _to_json_bytes(self) -> bytes:
        return self._json_str.encode('utf-8')

    def constrain(self, z: List[int]) -> 'ConstrainedKey':
        """
        Derives a constrained key.
        :param z: List of large integers.
        :return: ConstrainedKey instance.
        """
        z_hex = [hex(v % self.modulus)[2:] for v in z]
        z_bytes = json.dumps(z_hex).encode('utf-8')
        res_bytes = _cprf_lib.C_Constrain(self._to_json_bytes(), z_bytes)
        return ConstrainedKey(res_bytes.decode('utf-8'))

    def eval(self, x: List[int]) -> bytes:
        """
        Evaluates the CPRF function with the master key.
        :param x: List of large integers.
        :return: A bytes object representing the hashed evaluation result.
        """
        x_hex = [hex(v % self.modulus)[2:] for v in x]
        x_bytes = json.dumps(x_hex).encode('utf-8')
        res_bytes = _cprf_lib.C_Eval(self._to_json_bytes(), x_bytes)
        return bytes.fromhex(res_bytes.decode('utf-8'))


def keygen(modulus: int, length: int) -> MasterKey:
    """
    Generates a new CPRF master key.
    
    :param modulus: An arbitrarily large integer representing the modulus.
    :param length: Length of the inner product space.
    :return: MasterKey instance.
    """
    modulus_hex = hex(modulus)[2:]
    res_bytes = _cprf_lib.C_KeyGen(modulus_hex.encode('utf-8'), length)
    return MasterKey(res_bytes.decode('utf-8'))

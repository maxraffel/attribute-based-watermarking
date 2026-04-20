package main

import (
	"C"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
)

// Master key for the CPRF
// length: length of the inner product
// modulus: inner product modulus
// z0: master key
type MasterKey struct {
	length  int
	modulus *big.Int
	z0      []*big.Int
}

// Constrained key for the CPRF
// length: length of the inner product
// modulus: inner product modulus
// z1: constrained key
type ConstrainedKey struct {
	length  int
	modulus *big.Int
	z1      []*big.Int
}

// KeyGen generates a new CPRF key
// modulus: inner product modulus
// length: length of the input vector
// Outputs a CPRF master key
func KeyGen(modulus *big.Int, length int) (*MasterKey, error) {

	msk := &MasterKey{}
	msk.modulus = modulus
	msk.length = length
	msk.z0 = make([]*big.Int, length)

	var err error

	for i := 0; i < length; i++ {
		msk.z0[i], err = generateRandomBigInt(modulus)
		if err != nil {
			return nil, fmt.Errorf("failed to generate master key component %d: %w", i, err)
		}
	}

	return msk, nil
}

// Constrain outputs a constrained key for the CPRF
func (msk *MasterKey) Constrain(z []*big.Int) (*ConstrainedKey, error) {

	length := msk.length
	modulus := msk.modulus

	csk := &ConstrainedKey{}
	csk.modulus = modulus
	csk.length = length
	csk.z1 = make([]*big.Int, length)

	delta, err := generateRandomBigInt(modulus)
	if err != nil {
		return nil, fmt.Errorf("failed to generate delta for constraint: %w", err)
	}

	// the constraint key is computed as z0 - z*Delta
	// for a random Delta
	for i := 0; i < length; i++ {
		csk.z1[i] = big.NewInt(0)
		csk.z1[i].Mul(delta, z[i])          // z*Delta
		csk.z1[i].Sub(msk.z0[i], csk.z1[i]) // z0 - z*Delta
		if err != nil {
			return nil, err
		}
	}

	return csk, nil
}

func (msk *MasterKey) Eval(x []*big.Int) []byte {
	modulus := msk.modulus
	length := msk.length
	return commonEval(modulus, length, msk.z0, x)
}

func (csk *ConstrainedKey) CEval(x []*big.Int) []byte {
	modulus := csk.modulus
	length := csk.length
	return commonEval(modulus, length, csk.z1, x)
}

func commonEval(
	modulus *big.Int,
	length int,
	zb []*big.Int,
	x []*big.Int) []byte {

	tmp := big.NewInt(0)
	k := big.NewInt(0) // inner product result
	for i := 0; i < length; i++ {
		tmp.Mul(zb[i], x[i])
		k.Add(k, tmp).Mod(k, modulus)
	}

	return hashSHA256(k, x)
}

// SHÁ256 as a collision-resistant hash function.
// k: PRF key
// x: input vector
func hashSHA256(k *big.Int, x []*big.Int) []byte {

	byteInput := make([]byte, 0)

	byteInput = append(byteInput, k.Bytes()...)

	for i := 0; i < len(x); i++ {
		byteInput = append(byteInput, x[i].Bytes()...)
	}

	hasher := sha256.New()
	hasher.Write(byteInput)
	hash := hasher.Sum(nil)

	return hash
}

func generateRandomBigInt(max *big.Int) (*big.Int, error) {
	randomInt, err := rand.Int(rand.Reader, max)
	if err != nil {
		return nil, fmt.Errorf("failed to generate random number: %w", err)
	}
	return randomInt, nil
}

// ---------------------------------------------------------
// C-GO PYTHON WRAPPERS
// ---------------------------------------------------------

type MasterKeyJSON struct {
	Length  int      `json:"length"`
	Modulus string   `json:"modulus"`
	Z0      []string `json:"z0"`
}

type ConstrainedKeyJSON struct {
	Length  int      `json:"length"`
	Modulus string   `json:"modulus"`
	Z1      []string `json:"z1"`
}

func (msk *MasterKey) toJSON() string {
	mj := MasterKeyJSON{
		Length:  msk.length,
		Modulus: msk.modulus.Text(16),
		Z0:      make([]string, len(msk.z0)),
	}
	for i, v := range msk.z0 {
		if v != nil {
			mj.Z0[i] = v.Text(16)
		}
	}
	b, _ := json.Marshal(mj)
	return string(b)
}

func mskFromJSON(s string) (*MasterKey, error) {
	var mj MasterKeyJSON
	if err := json.Unmarshal([]byte(s), &mj); err != nil {
		return nil, err
	}
	modulus := new(big.Int)
	modulus.SetString(mj.Modulus, 16)
	z0 := make([]*big.Int, len(mj.Z0))
	for i, v := range mj.Z0 {
		z0[i] = new(big.Int)
		z0[i].SetString(v, 16)
	}
	return &MasterKey{length: mj.Length, modulus: modulus, z0: z0}, nil
}

func (csk *ConstrainedKey) toJSON() string {
	cj := ConstrainedKeyJSON{
		Length:  csk.length,
		Modulus: csk.modulus.Text(16),
		Z1:      make([]string, len(csk.z1)),
	}
	for i, v := range csk.z1 {
		if v != nil {
			cj.Z1[i] = v.Text(16)
		}
	}
	b, _ := json.Marshal(cj)
	return string(b)
}

func cskFromJSON(s string) (*ConstrainedKey, error) {
	var cj ConstrainedKeyJSON
	if err := json.Unmarshal([]byte(s), &cj); err != nil {
		return nil, err
	}
	modulus := new(big.Int)
	modulus.SetString(cj.Modulus, 16)
	z1 := make([]*big.Int, len(cj.Z1))
	for i, v := range cj.Z1 {
		z1[i] = new(big.Int)
		z1[i].SetString(v, 16)
	}
	return &ConstrainedKey{length: cj.Length, modulus: modulus, z1: z1}, nil
}

//export C_KeyGen
func C_KeyGen(modulusHex *C.char, length C.int) *C.char {
	modStr := C.GoString(modulusHex)
	modulus := new(big.Int)
	modulus.SetString(modStr, 16)

	msk, err := KeyGen(modulus, int(length))
	if err != nil {
		return C.CString("")
	}
	return C.CString(msk.toJSON())
}

//export C_Constrain
func C_Constrain(mskJson *C.char, zJsonArray *C.char) *C.char {
	mskStr := C.GoString(mskJson)
	zStr := C.GoString(zJsonArray)

	msk, err := mskFromJSON(mskStr)
	if err != nil {
		return C.CString("")
	}

	var zHex []string
	if err := json.Unmarshal([]byte(zStr), &zHex); err != nil {
		return C.CString("")
	}

	z := make([]*big.Int, len(zHex))
	for i, v := range zHex {
		z[i] = new(big.Int)
		z[i].SetString(v, 16)
	}

	csk, err := msk.Constrain(z)
	if err != nil {
		return C.CString("")
	}
	return C.CString(csk.toJSON())
}

//export C_Eval
func C_Eval(mskJson *C.char, xJsonArray *C.char) *C.char {
	mskStr := C.GoString(mskJson)
	xStr := C.GoString(xJsonArray)

	msk, err := mskFromJSON(mskStr)
	if err != nil {
		return C.CString("")
	}

	var xHex []string
	if err := json.Unmarshal([]byte(xStr), &xHex); err != nil {
		return C.CString("")
	}

	x := make([]*big.Int, len(xHex))
	for i, v := range xHex {
		x[i] = new(big.Int)
		x[i].SetString(v, 16)
	}

	out := msk.Eval(x)
	return C.CString(hex.EncodeToString(out))
}

//export C_CEval
func C_CEval(cskJson *C.char, xJsonArray *C.char) *C.char {
	cskStr := C.GoString(cskJson)
	xStr := C.GoString(xJsonArray)

	csk, err := cskFromJSON(cskStr)
	if err != nil {
		return C.CString("")
	}

	var xHex []string
	if err := json.Unmarshal([]byte(xStr), &xHex); err != nil {
		return C.CString("")
	}

	x := make([]*big.Int, len(xHex))
	for i, v := range xHex {
		x[i] = new(big.Int)
		x[i].SetString(v, 16)
	}

	out := csk.CEval(x)
	return C.CString(hex.EncodeToString(out))
}

func main() {}

import prc
import cprf
import secrets
from hashlib import sha256


print("\n--- Testing Many Random Strings ---")
TEST_CODE_LEN = 1200
prc.set_code_length(TEST_CODE_LEN)
random_testing_s = prc.key_gen_from_seed(sha256(secrets.token_bytes(TEST_CODE_LEN)).digest())
num_random_tests = 5000
false_positives = 0
total_ber = 0.0
secret_int = secrets.randbits(TEST_CODE_LEN)
for _ in range(num_random_tests):
    rand_int = secrets.randbits(TEST_CODE_LEN)
    ber = (secret_int ^ rand_int).bit_count() / TEST_CODE_LEN
    total_ber += ber
        
    # Generate boolean bits from integer representation
    bits = [c == '1' for c in bin(rand_int)[2:].zfill(TEST_CODE_LEN)]
    if prc.detect(random_testing_s, bits):
        false_positives += 1
            
avg_ber = total_ber / num_random_tests
print(f"Tested {num_random_tests} random strings.")
print(f"Average BER: {avg_ber:.2%}")
print(f"False Positives (Watermark Detected): {false_positives} / {num_random_tests} ({false_positives/num_random_tests:.2%})")

print("\n--- Testing CPRF + PRC Pipeline ---")
CODE_LEN = 12000
prc.set_code_length(CODE_LEN)
sk = cprf.keygen(1024, CODE_LEN)

num_cases = 100
expected_success = 0
expected_failure = 0
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

for i in range(num_cases):
    # To test varying outcomes, we create random f and x vectors
    x = [secrets.randbelow(10) for _ in range(CODE_LEN)]
    f = [secrets.randbelow(10) for _ in range(CODE_LEN)]
    
    # Force 50% of the cases to succeed by forcing inner product to be 0 mod 1024
    if i % 2 == 0:
        # Since 1024 = 2^10, any ODD number is coprime to 1024
        x[-1] = secrets.randbelow(5) * 2 + 1 
        current_sum = sum(f[j] * x[j] for j in range(CODE_LEN - 1))
        inv = pow(x[-1], -1, 1024)
        f[-1] = ( -current_sum * inv ) % 1024
    
    inner_product = sum(f[j] * x[j] for j in range(CODE_LEN)) % 1024
    expected_to_succeed = (inner_product == 0)
    
    if expected_to_succeed:
        expected_success += 1
    else:
        expected_failure += 1
        
    dk = sk.constrain(f)
    
    # Issue a real watermark for x
    r = sk.eval(x)
    s = prc.key_gen_from_seed(sha256(r).digest())
    c = prc.encode(s)
    secret_bits = [bool(b) for b in c]
    
    # Attempt detect
    recovered_r = dk.c_eval(x)
    recovered_s = prc.key_gen_from_seed(sha256(recovered_r).digest())
    
    actually_succeeded = prc.detect(recovered_s, secret_bits)
    
    if expected_to_succeed and actually_succeeded:
        true_positive += 1
    elif expected_to_succeed and not actually_succeeded:
        false_negative += 1
    elif not expected_to_succeed and not actually_succeeded:
        true_negative += 1
    elif not expected_to_succeed and actually_succeeded:
        false_positive += 1

print(f"Total Cases: {num_cases}")
print(f"Expected to succeed: {expected_success}")
print(f"  Actually succeeded (True Positives): {true_positive}")
print(f"  Fails despite expected (False Negatives): {false_negative}")
print(f"Expected to fail: {expected_failure}")
print(f"  Actually fails (True Negatives): {true_negative}")
print(f"  Succeeds despite expected fail (False Positives): {false_positive}")
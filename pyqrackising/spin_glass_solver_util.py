from numba import njit


@njit
def get_cut_from_bit_array(theta, nodes):
    bitstring = ""
    l, r = [], []
    for i in range(len(theta)):
        b = theta[i]
        if b:
            bitstring += "1"
            r.append(nodes[i])
        else:
            bitstring += "0"
            l.append(nodes[i])

    return bitstring, l, r

# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]

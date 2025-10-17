from sympy import symbols, cos, sin, Matrix, simplify, Basic, latex, N
from typing import Optional, Dict, Any
import math

def print_sympy_matrix(M: Matrix,
                        subs: Optional[Dict[Basic, Any]] = None,
                        float_digits: int = 4,
                        max_entry_len: Optional[int] = 200,
                        use_latex: bool = False) -> str:
    """
    Return a nicely formatted string for a SymPy Matrix.
    
    Args:
        M: SymPy Matrix to format.
        subs: optional dict of substitutions (e.g. {theta: 1.2, thdot: 0.1}).
        float_digits: number of digits to show for numeric entries.
        max_entry_len: truncate any stringified element longer than this with '...'.
        use_latex: if True, return LaTeX representation (ignore other formatting).
    
    Returns:
        A multi-line string with a neat, column-aligned matrix representation
        (or LaTeX string if use_latex=True).
    """
    if use_latex:
        # Return latex form for copy/paste into notebook or report
        if subs:
            return latex(M.subs(subs))
        return latex(M)

    # Apply substitutions if provided
    if subs:
        M_eval = M.subs(subs)
    else:
        M_eval = M

    # Convert each element to a readable string
    rows, cols = M_eval.shape
    str_matrix = [['' for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            e = simplify(M_eval[i, j])
            # If no symbolic variables left -> try numeric formatting
            if not e.free_symbols:
                try:
                    # Use numeric approximation if it's a rational or float
                    val = float(N(e, float_digits + 2))
                    entry = f"{val:.{float_digits}f}"
                except Exception:
                    entry = str(e)
            else:
                entry = str(e)  # keep symbolic form

            # shorten overly long elements
            if max_entry_len and len(entry) > max_entry_len:
                entry = entry[:max_entry_len-3] + "..."
            str_matrix[i][j] = entry

    # Compute column widths and build aligned rows
    col_widths = [max(len(str_matrix[r][c]) for r in range(rows)) for c in range(cols)]
    row_strings = []
    for i in range(rows):
        parts = []
        for j in range(cols):
            parts.append(str_matrix[i][j].rjust(col_widths[j]))
        row_strings.append("  ".join(parts))

    # Build bracketed matrix block
    # e.g. [ a  b ]
    #      [ c  d ]
    left_bracket = "[ "
    right_bracket = " ]"
    formatted_lines = []
    for idx, rstr in enumerate(row_strings):
        formatted_lines.append(f"{left_bracket if idx==0 else '  '}{rstr}{right_bracket if idx==0 else ''}")

    # Add proper top/bottom brackets visually
    # (we'll do a compact visual with first line including left bracket and others aligned)
    return "\n".join(formatted_lines)

def se3_to_SE3(M: Matrix):
    SE3 = Matrix([[0,    -M[2, 0], M[1, 0]],
                  [M[2, 0],  0,   -M[0, 0]],
                  [-M[1, 0], M[0, 0], 0]])
    return SE3

if __name__ == "__main__":
    # symbols
    theta, beta, tdot, bdot, s40, c40= symbols('theta beta tdot bdot s40 c40', real=True)

    # rot(z,beta)
    A = Matrix([[cos(beta), -sin(beta), 0],
                [sin(beta),  cos(beta), 0],
                [0, 0, 1]])

    # rot(x,40)
    B = Matrix([[1, 0, 0],
                [0, c40, -s40],
                [0, s40,  c40]])

    # rot(y,theta)
    C = Matrix([[cos(theta), 0, sin(theta)],
                [0, 1, 0],
                [-sin(theta), 0, cos(theta)]])

    # find A dot
    Adot = simplify(A.diff(beta) * bdot)
    # Find A dotdot
    Addot = simplify(A.diff(beta).diff(beta) * bdot**2)

    # R = A * B * C
    R = simplify(A * B * C)

    # parameter vector and time-derivative vector
    q = Matrix([theta, beta])
    qdot = Matrix([tdot, bdot])

    # flatten R to 9x1 and compute 9x2 Jacobian
    R_vec = Matrix(R).reshape(9, 1)    # column vector of entries
    J = R_vec.jacobian(q)              # 9 x 2 Jacobian

    # time derivative of flattened R
    Rdot_vec = simplify(J * qdot)      # 9x1

    # second time derivative (qddot = 0 since tdot, bdot constants): Rddot = Jdot * qdot
    Jdot = simplify(J.diff(theta) * tdot + J.diff(beta) * bdot)
    Rddot_vec = simplify(Jdot * qdot)

    # reshape back to 3x3
    Rdot = Matrix(Rdot_vec).reshape(3, 3)
    Rddot = Matrix(Rddot_vec).reshape(3, 3)

    # Pretty print symbolic matrices
    print("R =")
    print(print_sympy_matrix(R))
    print("\nAdot (symbolic) =")
    print(print_sympy_matrix(Adot))
    print("\nR_dot (symbolic) =")
    print(print_sympy_matrix(Rdot))
    print("\nR_ddot (symbolic) =")
    print(print_sympy_matrix(Rddot))

    # Example with numerical substitutions in radians
    subs_vals = {theta: 0.0, beta: 0.0, tdot: 6.28, bdot: 3.0, s40: math.sin(40/180*math.pi), c40: math.cos(40/180*math.pi)}
    print("\nAdot (numeric) =")
    print(print_sympy_matrix(Adot, subs=subs_vals, float_digits=6))
    P1 = Matrix([0, 350, 150])
    print("\nAdot*P1 (numeric) =")
    print(print_sympy_matrix(Adot*P1, subs=subs_vals, float_digits=6))
    print("\nAddot (numeric) =")
    print(print_sympy_matrix(Addot, subs=subs_vals, float_digits=6))
    print("\nAddot*P1 (numeric) =")
    print(print_sympy_matrix(Addot*P1, subs=subs_vals, float_digits=6))

    print("\nRdot (numeric) =")
    print(print_sympy_matrix(Rdot, subs=subs_vals, float_digits=6))
    P2 = Matrix([0, 300, 120])
    print("\nRdot*P2 (numeric) =")
    print(print_sympy_matrix(Rdot*P2, subs=subs_vals, float_digits=6))
    print("\nRddot (numeric) =")
    print(print_sympy_matrix(Rddot, subs=subs_vals, float_digits=6))
    print("\nRddot*P2 (numeric) =")
    print(print_sympy_matrix(Rddot*P2, subs=subs_vals, float_digits=6))
    
    print ("\nVerification by Lie's algebra:")
    w0 = Matrix([0, 0, 3]) # Expressed in frame 0
    w1 = Matrix([0, 6.28, 0]) # Expressed in frame 1

    W0 = se3_to_SE3(w0) # Expressed in frame 0
    W1 = se3_to_SE3(w0 + A * B * w1) # Expressed in frame 0

    # find W1 dot
    print("\nW1 (symbolic)=")
    print(print_sympy_matrix(W1, subs=subs_vals, float_digits=6))

    W1_dot = simplify(W1.diff(beta) * bdot) # As it only affect by beta
    print("\nW1_dot (symbolic)=")
    print(print_sympy_matrix(W1_dot, subs=subs_vals, float_digits=6))

    print("\nAdot (verification)=")
    print(print_sympy_matrix(W0*A, subs=subs_vals, float_digits=6))
    print("\nRdot (verification)=")
    print(print_sympy_matrix(W1*R, subs=subs_vals, float_digits=6))

    print("\nAddot (verification)=")
    print(print_sympy_matrix(W0*W0*A, subs=subs_vals, float_digits=6))
    print("\nRddot (verification)=")
    print(print_sympy_matrix(W1_dot*R+W1*W1*R, subs=subs_vals, float_digits=6))
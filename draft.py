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

    # Allow scalar SymPy expressions by wrapping them as 1x1 matrix
    if not hasattr(M_eval, 'shape'):
        M_eval = Matrix([[M_eval]])

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



    w0 = Matrix([0, 0, 3])
    w1 = Matrix([0, 4.81, 7.03])

    W0 = se3_to_SE3(w0)
    W1 = se3_to_SE3(w1)

    print("W0 =")
    print(print_sympy_matrix(W0))
    print("W1 =")
    print(print_sympy_matrix(W1))


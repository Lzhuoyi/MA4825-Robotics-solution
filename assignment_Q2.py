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

if __name__ == "__main__":
    # joint symbols
    theta1, theta2, theta3, S4, d_theta1, d_theta2, d_theta3, d_S4 = symbols('theta1 theta2 theta3 S4 d_theta1 d_theta2 d_theta3 d_S4', real=True)
    # link symbols
    L1, a1, L2, L3, b4 = symbols('L1 a1 L2 L3 b4', real=True)

    # **************** Rotation Matices ******************* #
    # rot(z,theta1)
    A = Matrix([[cos(theta1), -sin(theta1), 0],
                [sin(theta1),  cos(theta1), 0],
                [0, 0, 1]])

    # rot(z,theta2)
    B = Matrix([[cos(theta2), -sin(theta2), 0],
                [sin(theta2),  cos(theta2), 0],
                [0, 0, 1]])

    # rot(z,theta3)
    C = Matrix([[cos(theta3), -sin(theta3), 0],
                [sin(theta3),  cos(theta3), 0],
                [0, 0, 1]])

    # R = A * B * C
    R_02 = simplify(A * B)
    R_03 = simplify(A * B * C)

    # Pretty print symbolic matrices
    print("R02 =")
    print(print_sympy_matrix(R_02))
    print("R03 =")
    print(print_sympy_matrix(R_03))


    # **************** Forward Kinematics ******************* #
    P1 = Matrix([0, L1, a1])
    P2 = Matrix([0, L2, 0])
    P3 = Matrix([0, -b4, -L3-S4])

    FK = simplify(A*P1 + R_02*P2 + R_03*P3)
    print("Forward Kinematics P0 =")
    print(print_sympy_matrix(FK))

    # **************** Jacobian ******************* #

    # parameter vector and time-derivative vector
    q = Matrix([theta1, theta2, theta3, S4])
    qdot = Matrix([d_theta1, d_theta2, d_theta3, d_S4])

    # compute Jacobian
    Jacobian = FK.jacobian(q)             

    # compute velocity
    velocity = simplify(Jacobian * qdot)      

    print("\nJacobian =")
    print(print_sympy_matrix(Jacobian))
    print("\nVelocity =")
    print(print_sympy_matrix(velocity))

    # **************** centroid velocity ******************* #

    # find R01 dot
    R_01 = A
    R_01vec = Matrix(R_01).reshape(9, 1)   # column vector of entries
    J = R_01vec.jacobian(q) 
    dR_01vec = simplify(J * qdot)     # time derivative of flattened R 9*1
    dR_01 = Matrix(dR_01vec).reshape(3, 3)   # reshape back to 3x3
    # centroid of link1:
    P_c1 = Matrix([0, L1/2, 0])
    print("dR_01 =")
    print(print_sympy_matrix(dR_01))
    print("V_c1 =")
    print(print_sympy_matrix(dR_01*P_c1))

    # find R02 dot
    R_02vec = Matrix(R_02).reshape(9, 1)   # column vector of entries
    J = R_02vec.jacobian(q) 
    dR_02vec = simplify(J * qdot)     # time derivative of flattened R 9*1
    dR_02 = Matrix(dR_02vec).reshape(3, 3)   # reshape back to 3x3
    # centroid of link1:
    P_c2 = Matrix([0, L2/2, 0])
    print("dR_02 =")
    print(print_sympy_matrix(dR_02))
    print("V_c2 =")
    print(print_sympy_matrix(dR_01*P1 + dR_02*P_c2))
    # v = dR_01*P1 + dR_02*P_c2
    # print(print_sympy_matrix(v.norm()))

    # find R03 dot
    R_03vec = Matrix(R_03).reshape(9, 1)   # column vector of entries
    J = R_03vec.jacobian(q) 
    dR_03vec = simplify(J * qdot)     # time derivative of flattened R 9*1
    dR_03 = Matrix(dR_03vec).reshape(3, 3)   # reshape back to 3x3
    # centroid of link1:
    P_c3 = Matrix([0, 0, -L3/2])
    print("dR_03 =")
    print(print_sympy_matrix(dR_03))
    print("V_c3 =")
    print(print_sympy_matrix(dR_01*P1 + dR_02*P2 + dR_03*P_c3))
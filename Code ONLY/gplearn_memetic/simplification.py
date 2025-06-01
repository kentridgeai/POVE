import itertools
import ast
import re

CONSTANT = 'C'  # Single constant token
OPERATORS = ['+', '-', '*', '/']
FUNCTIONS = []

class SimplificationError(Exception):
    """Custom exception for errors encountered during expression simplification."""
    pass

# Function to count the number of tokens in an AST expression
def count_tokens(node):
    """Counts the number of tokens (variables, constants, operators, and functions) in an AST tree."""
    if isinstance(node, ast.BinOp):
        return 1 + count_tokens(node.left) + count_tokens(node.right)  # Each BinOp counts as one operator
    elif isinstance(node, ast.Call):
        return 1 + sum(count_tokens(arg) for arg in node.args)  # Function call + argument tokens
    elif isinstance(node, ast.Name):  # Variable or constant
        return 1
    return 0  # Ignore unsupported nodes

# Function to count the expression length
def count_expression_length(expr):
    return count_tokens(ast.parse(expr, mode='eval').body)

def flatten_add(node):
    """
    Given a simplified AST node for addition, recursively flatten it so that
    all operands of '+' appear in a single list.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return flatten_add(node.left) + flatten_add(node.right)
    else:
        return [node]

def flatten_mult(node):
    """
    Given a simplified AST node for multiplication, recursively flatten it so that
    all operands of '*' appear in a single list.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return flatten_mult(node.left) + flatten_mult(node.right)
    else:
        return [node]

def is_constant(node):
    """Return True if node represents the constant token."""
    return isinstance(node, ast.Name) and node.id == CONSTANT

def is_one(node):
    """Return True if node represents the constant token."""
    return isinstance(node, ast.Constant) and node.value == 1

def is_zero(node):
    """Return True if node represents the constant token."""
    return isinstance(node, ast.Constant) and node.value == 0

# Function to simplify an expression (single function for all transformations)
def simplify_expression(expr):
    """
    Implements:
    - **Tree Reordering**: Ensures commutative operations follow a fixed order.
    - **Reparameterization Invariance**: Simplifies constant expressions and removes redundant function calls.
    Uses AST parsing to enforce these transformations.
    """
    try:
        tree = ast.parse(expr, mode='eval')
        simplified_ast = tree.body
        for i in range(2):
            simplified_ast = simplify_ast(simplified_ast)
        return ast_to_string(simplified_ast)
    except Exception as e:
        raise SimplificationError(f"Error simplifying expression '{expr}': {e}")

# Simplify AST for all transformations
def simplify_ast(node):
    """
    Applies:
    - **Tree Reordering**: Ensures commutative operations follow a fixed order.
    - **Reparameterization Invariance**: Collapses redundant constant function calls.
    """

    if isinstance(node, ast.BinOp):
        left = simplify_ast(node.left)
        right = simplify_ast(node.right)

        # --- Cancellation for subtraction ---
        if isinstance(node.op, ast.Sub):
            if ast.dump(left) == ast.dump(right):
                return ast.Constant(value=0)

        # --- Cancellation for division ---
        if isinstance(node.op, ast.Div):
            if ast.dump(left) == ast.dump(right):
                return ast.Constant(value=1)

        # Reparameterization Invariance: Collapse C + C = C, C * C = C, C - C = C, C / C = C
        if isinstance(left, ast.Name) and isinstance(right, ast.Name) and left.id == right.id == CONSTANT:
            return left

        # Tree Reordering: Ensure commutative operations are ordered consistently
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = sorted([left, right], key=ast.dump)

        # **Hardcoded Simplifications for Addition**
        if isinstance(node.op, ast.Add):
            if isinstance(left, ast.BinOp) and isinstance(left.op, (ast.Add, ast.Sub)) and is_constant(right):
                if is_constant(left.left) or is_constant(left.right):  # (C + x) + C → C + x, (C - x) + C → C - x
                    node = left
            elif isinstance(right, ast.BinOp) and isinstance(right.op, (ast.Add, ast.Sub)) and is_constant(left):
                if is_constant(right.left) or is_constant(right.right):  # C + (C + x) → C + x, C + (C - x) → C - x
                    node = right
            operands = flatten_add(node)
            # Merge constant tokens: count how many times a constant appears.
            constant_count = sum(1 for opnd in operands if is_constant(opnd))
            # Remove all constant tokens and zeroes.
            operands = [simplify_ast(opnd) for opnd in operands]
            operands = [opnd for opnd in operands if not is_zero(opnd)]
            operands = [opnd for opnd in operands if not is_constant(opnd)]
            if not operands:
                return ast.Name(id=CONSTANT, ctx=ast.Load())
            # If any constant appeared, add back exactly one constant token.
            if constant_count > 0:
                operands.append(ast.Name(id=CONSTANT, ctx=ast.Load()))
            # Optionally sort operands to get a canonical ordering.
            operands = sorted(operands, key=ast.dump)
            # Rebuild the left-associated addition tree.
            result = operands[-1]
            for opnd in operands[:-1]:
                left, right = sorted([result, opnd], key=ast.dump)
                result = ast.BinOp(left=left, op=ast.Add(), right=right)
            return result

        # **Hardcoded Simplifications for Multiplication**
        elif isinstance(node.op, ast.Mult):
            if isinstance(left, ast.BinOp) and isinstance(left.op, (ast.Mult,ast.Div)) and is_constant(right):
                if is_constant(left.left) or is_constant(left.right):  # (C * x) * C → C * x, (C / x) * C → C / x
                    node = left
            elif isinstance(right, ast.BinOp) and isinstance(right.op, (ast.Mult,ast.Div)) and is_constant(left):
                if is_constant(right.left) or is_constant(right.right):  # C * (C * x) → C * x, C * (C / x) → C / x
                    node = right         
            operands = flatten_mult(node)
            # Merge constant tokens: count how many times a constant appears.
            constant_count = 0
            for opnd in operands:
                if is_constant(opnd):
                    constant_count+=1
                elif is_zero(opnd):
                    return ast.Constant(value=0)
            # Remove all constant tokens and ones.
            operands = [simplify_ast(opnd) for opnd in operands]
            operands = [opnd for opnd in operands if not is_one(opnd)]
            operands = [opnd for opnd in operands if not is_constant(opnd)]
            if not operands:
                return ast.Name(id=CONSTANT, ctx=ast.Load())
            # If any constant appeared, add back exactly one constant token.
            if constant_count > 0:
                operands.append(ast.Name(id=CONSTANT, ctx=ast.Load()))
            # Optionally sort operands to get a canonical ordering.
            operands = sorted(operands, key=ast.dump)
            # Rebuild the left-associated multiplication tree.
            result = operands[-1]
            for opnd in operands[:-1]:
                left, right = sorted([result, opnd], key=ast.dump)
                result = ast.BinOp(left=left, op=ast.Mult(), right=right)
            return result

        # **Hardcoded Simplifications for Subtraction**
        elif isinstance(node.op, ast.Sub):
            if is_constant(right): # x - C → x + C
                return simplify_ast(ast.BinOp(left=left, op=ast.Add(), right=right))
            elif is_zero(right):
                return left
            elif isinstance(right, ast.BinOp) and isinstance(right.op, ast.Add) and is_constant(left):
                if is_constant(right.left):  # C - (C + x) → C - x
                    return (ast.BinOp(left=right.left, op=ast.Sub(), right=right.right))
                if is_constant(right.right):  # C - (x + C) → C - x
                    return (ast.BinOp(left=right.right, op=ast.Sub(), right=right.left))
            elif isinstance(right, ast.BinOp) and isinstance(right.op, ast.Sub) and is_constant(left):
                if is_constant(right.left):  # C - (C - x) → C + x
                    return simplify_ast(ast.BinOp(left=right.left, op=ast.Add(), right=right.right))
                if is_constant(right.right):  # C - (x - C) → C - x (should not be reached because x - C → x + C should occur first)
                    return (ast.BinOp(left=right.right, op=ast.Sub(), right=right.left))

        # **Hardcoded Simplifications for Division**
        elif isinstance(node.op, ast.Div):
            if is_constant(right): # x / C → x * C
                return simplify_ast(ast.BinOp(left=left, op=ast.Mult(), right=right))
            elif is_one(right) or is_zero(left):
                return left
            elif isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult) and is_constant(left):
                if is_constant(right.left):  # C / (C * x) → C / x
                    return (ast.BinOp(left=right.left, op=ast.Div(), right=right.right))
                if is_constant(right.right):  # C / (x * C) → C / x
                    return (ast.BinOp(left=right.right, op=ast.Div(), right=right.left))
            elif isinstance(right, ast.BinOp) and isinstance(right.op, ast.Div) and is_constant(left):
                if is_constant(right.left):  # C / (C / x) → C * x
                    return simplify_ast(ast.BinOp(left=right.left, op=ast.Mult(), right=right.right))
                if is_constant(right.right):  # C / (x / C) → C / x (should not be reached because x / C → x * C will occur first)
                    return (ast.BinOp(left=right.right, op=ast.Div(), right=right.left))

        return ast.BinOp(left=left, op=node.op, right=right)

    elif isinstance(node, ast.Call):
        # Ensure function calls are recognized properly
        func = getattr(node.func, "id", None)  # Extract function name safely
        if func and func in FUNCTIONS:  # Only allow valid symbolic functions
            args = sorted([simplify_ast(arg) for arg in node.args], key=ast.dump)

            # Reparameterization Invariance: sin(C) = C
            if all(isinstance(arg, ast.Name) and arg.id == CONSTANT for arg in args):
                return args[0]  # sin(C) -> C

            return ast.Call(func=node.func, args=args, keywords=[])

    elif isinstance(node, ast.Name):
        return node

    elif isinstance(node, ast.Constant):
        return node

    else:
        raise SimplificationError(f"Unsupported AST node type: {type(node)} in expression.")

# Convert AST back to string representation
def ast_to_string(node):
    """Converts an AST node back into its canonical string form."""
    if isinstance(node, ast.BinOp):
        left = ast_to_string(node.left)
        right = ast_to_string(node.right)
        op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
        return f"({left} {op} {right})"

    elif isinstance(node, ast.Call):
        func = getattr(node.func, "id", None)  # Extract function name
        if func and func in FUNCTIONS:
            args = ', '.join(ast_to_string(arg) for arg in node.args)
            return f"{func}({args})"

    elif isinstance(node, ast.Name):
        return node.id

    elif isinstance(node, ast.Constant):
        return str(node.value)
        
    else:
        raise SimplificationError(f"Unsupported AST node type: {type(node)} when converting to string.")

def replace_math_functions(expression):
    # Define function replacements in order of operation precedence
    replacements = {
        r'add\(([^,]+),\s*([^\)]+)\)': r'(\1 + \2)',
        r'sub\(([^,]+),\s*([^\)]+)\)': r'(\1 - \2)',
        r'mul\(([^,]+),\s*([^\)]+)\)': r'(\1 * \2)',
        r'div\(([^,]+),\s*([^\)]+)\)': r'(\1 / \2)'
    }
    
    # Apply replacements iteratively until no more matches are found
    while any(re.search(pattern, expression) for pattern in replacements):
        for pattern, repl in replacements.items():
            expression = re.sub(pattern, repl, expression)
    
    return expression

def replace_constants(expression):
    # Replace all numeric constants (including decimals and negatives) with 'C'
    expression = re.sub(r'(?<![a-zA-Z_])[-+]?(?:\d*\.\d+|\d+)\b', 'C', expression)
    return expression

def replace_C_with_indices(expression):
    C_count = expression.count('C')
    for i in range(C_count):
        expression = expression.replace('C', f'D[{i}]', 1)
    expression = expression.replace('D', 'C')
    return expression, C_count

def replace_C_indices_with_values(expression, C_values):
    """Replaces C[i] in the expression with its corresponding value from C_values."""
    for i, value in enumerate(C_values):
        expression = expression.replace(f'C[{i}]', str(value))
    return expression

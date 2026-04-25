
import re

def fix_indentation_and_pass(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. Fix "except ...: pass" followed by indented code
        if re.match(r'^\s+except.*:\s*$', line):
            except_indent = len(line) - len(line.lstrip())
            new_lines.append(line)
            i += 1
            
            # Skip any incorrectly indented 'pass' lines immediately following 'except'
            while i < len(lines) and lines[i].strip() == 'pass':
                line_indent = len(lines[i]) - len(lines[i].lstrip())
                # If pass is not at except_indent + 4, it's likely misplaced
                if line_indent != except_indent + 4:
                    i += 1
                else:
                    break
            
            # Now we are at the first real statement after 'except' (or a correctly indented pass)
            # If the next line is more indented than except_indent, we should ensure it's exactly +4
            first_statement_idx = i
            while i < len(lines) and not lines[i].strip(): # skip empty lines
                i += 1
            
            if i < len(lines):
                stmt_line = lines[i]
                stmt_indent = len(stmt_line) - len(stmt_line.lstrip())
                if stmt_indent > except_indent and stmt_indent != except_indent + 4:
                    # Fix this and subsequent lines until we hit a line with <= except_indent
                    diff = (except_indent + 4) - stmt_indent
                    while i < len(lines):
                        if not lines[i].strip():
                            new_lines.append(lines[i])
                        else:
                            curr_indent = len(lines[i]) - len(lines[i].lstrip())
                            if curr_indent <= except_indent:
                                break
                            new_lines.append(' ' * (curr_indent + diff) + lines[i].lstrip())
                        i += 1
                    continue # i is already advanced
            
            # If no statement was found, add a 'pass'
            if first_statement_idx == i:
                 new_lines.append(' ' * (except_indent + 4) + 'pass\n')
                 
        else:
            new_lines.append(line)
            i += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

fix_indentation_and_pass('src/execution/trader_futures.py')
fix_indentation_and_pass('src/execution/trader_spot.py')

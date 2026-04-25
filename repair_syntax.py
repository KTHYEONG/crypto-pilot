
import re

def fix_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match "except ...:"
        if re.match(r'^\s+except.*:\s*$', line):
            new_lines.append(line)
            if i + 1 < len(lines) and lines[i+1].strip() == 'pass':
                # Check if the next-next line is more indented than 'except'
                except_indent = len(line) - len(line.lstrip())
                if i + 2 < len(lines):
                    next_next_line = lines[i+2]
                    next_next_indent = len(next_next_line) - len(next_next_line.lstrip())
                    if next_next_indent > except_indent:
                        # This 'pass' is likely misplaced. Skip it.
                        i += 1 
        else:
            new_lines.append(line)
        i += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

fix_file('src/execution/trader_futures.py')
fix_file('src/execution/trader_spot.py')

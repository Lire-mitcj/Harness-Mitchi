import sqlparse
import re

def rewrite_query_source(current_sql: str, target_source: str, replaces_objects: list[str]) -> str:
    parsed = sqlparse.parse(current_sql)[0]
    tokens = list(parsed.tokens)
    
    from_idx = -1
    for idx, token in enumerate(tokens):
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
            from_idx = idx
            break
            
    if from_idx == -1:
        return current_sql
        
    main_table_idx = -1
    for j in range(from_idx + 1, len(tokens)):
        t = tokens[j]
        if t.is_whitespace:
            continue
        main_table_idx = j
        break
        
    if main_table_idx == -1:
        return current_sql
        
    main_table_token = tokens[main_table_idx]
    alias = main_table_token.get_alias() if hasattr(main_table_token, 'get_alias') else None
    
    new_val = f"{target_source} {alias}" if alias else target_source
    new_token = sqlparse.sql.Token(sqlparse.tokens.Name, new_val)
    
    replaces_lower = {obj.lower() for obj in replaces_objects}
    
    end_idx = len(tokens)
    for j in range(main_table_idx + 1, len(tokens)):
        t = tokens[j]
        val = t.value.upper().strip()
        if isinstance(t, sqlparse.sql.Where) or val.startswith("WHERE"):
            end_idx = j
            break
        if t.ttype is sqlparse.tokens.Keyword or t.ttype is sqlparse.tokens.Keyword.DML:
            if val in ("GROUP", "ORDER", "LIMIT", "UNION", "HAVING", ";", "SELECT", "INSERT", "UPDATE", "DELETE"):
                end_idx = j
                break
            if any(val.startswith(kw) for kw in ("GROUP", "ORDER", "LIMIT", "UNION", "HAVING")):
                end_idx = j
                break
                
    j = main_table_idx + 1
    middle_tokens = []
    removed_aliases = set()
    
    while j < end_idx:
        t = tokens[j]
        val = t.value.upper().strip()
        if "JOIN" in val and (t.ttype is sqlparse.tokens.Keyword or t.ttype is None):
            join_end = end_idx
            for k in range(j + 1, end_idx):
                tk = tokens[k]
                tk_val = tk.value.upper().strip()
                if "JOIN" in tk_val and (tk.ttype is sqlparse.tokens.Keyword or tk.ttype is None):
                    join_end = k
                    break
            
            joined_table_name = ""
            joined_table_alias = ""
            for k in range(j + 1, join_end):
                tk = tokens[k]
                if isinstance(tk, sqlparse.sql.Identifier):
                    real_name = tk.get_real_name()
                    joined_table_name = real_name or tk.value
                    joined_table_alias = tk.get_alias()
                    break
                elif tk.ttype is None and tk.value.strip() and not tk.is_whitespace:
                    joined_table_name = tk.value
                    # Maybe it has no alias, or the alias is a separate token
                    break
            
            cleaned_joined_table = joined_table_name.strip().split('.')[-1].split()[0].replace('"', '').replace("'", "").replace("`", "").lower()
            
            if cleaned_joined_table in replaces_lower:
                if joined_table_alias:
                    removed_aliases.add(joined_table_alias)
                j = join_end
            else:
                middle_tokens.extend(tokens[j:join_end])
                j = join_end
        else:
            middle_tokens.append(t)
            j += 1
            
    print("Removed aliases:", removed_aliases)
    
    # Identify target_alias
    target_alias = alias or target_source
    
    # Print SELECT items and output names
    select_idx = -1
    for idx, token in enumerate(tokens):
        if token.ttype is sqlparse.tokens.Keyword.DML and token.value.upper() == "SELECT":
            select_idx = idx
            break
            
    def get_select_items(tokens_list, s_idx, f_idx):
        res_items = []
        for idx in range(s_idx + 1, f_idx):
            token = tokens_list[idx]
            if token.is_whitespace:
                continue
            if isinstance(token, sqlparse.sql.IdentifierList):
                for ident in token.get_identifiers():
                    res_items.append(ident)
            elif isinstance(token, sqlparse.sql.Identifier):
                res_items.append(token)
            elif token.ttype is None and token.value.strip() and token.value.strip() != ',':
                res_items.append(token)
        return res_items
        
    def get_output_name(item):
        if hasattr(item, 'get_alias') and item.get_alias():
            return item.get_alias().replace('"', '').replace("'", "").replace("`", "").strip()
        if hasattr(item, 'get_real_name') and item.get_real_name():
            return item.get_real_name().replace('"', '').replace("'", "").replace("`", "").strip()
        val = item.value.strip()
        parts = val.split('.')
        return parts[-1].replace('"', '').replace("'", "").replace("`", "").strip()

    select_items = get_select_items(tokens, select_idx, from_idx)
    print("\n--- SELECT ITEMS ---")
    for item in select_items:
        print(f"Item: {repr(item.value)} -> Output Name: {repr(get_output_name(item))}")
        
    # Get the old main table name if any to also replace references to it
    old_main_table_name = ""
    if isinstance(main_table_token, sqlparse.sql.Identifier):
        old_main_table_name = main_table_token.get_real_name() or main_table_token.value
    elif main_table_token.ttype is None and main_table_token.value.strip():
        old_main_table_name = main_table_token.value.strip()
    
    # Recursive function to rewrite identifiers
    def rewrite_identifiers(token, legacy_aliases, target_alias):
        if not hasattr(token, 'tokens') or not token.tokens:
            return
        
        if isinstance(token, sqlparse.sql.Identifier):
            sub = token.tokens
            if len(sub) >= 3:
                # Check if first token is name-like, and second is a dot
                first_t = sub[0]
                second_t = sub[1]
                if (first_t.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder) or first_t.ttype is None) and second_t.value == '.':
                    prefix = first_t.value.strip().replace('`','').replace('"','').replace("'", "").lower()
                    if prefix in legacy_aliases:
                        if target_alias:
                            # Modify the first token's value
                            first_t.value = target_alias
                        else:
                            # Strip the alias and dot
                            token.tokens = sub[2:]
                            
        for sub_token in token.tokens:
            rewrite_identifiers(sub_token, legacy_aliases, target_alias)
            
    # Lowercase legacy aliases for comparison
    legacy_lower = {a.lower() for a in removed_aliases if a}
    for obj in replaces_objects:
        legacy_lower.add(obj.lower())
    if old_main_table_name:
        legacy_lower.add(old_main_table_name.lower())
        
    # Recursive helper to check if a token references any legacy aliases
    def references_aliases(token, replaced_aliases):
        if isinstance(token, sqlparse.sql.Identifier):
            sub = token.tokens
            if len(sub) >= 2 and sub[1].value == '.':
                prefix = sub[0].value.strip().replace('`','').replace('"','').replace("'", "").lower()
                if prefix in replaced_aliases:
                    return True
        if hasattr(token, 'tokens') and token.tokens:
            return any(references_aliases(sub, replaced_aliases) for sub in token.tokens)
        return False

    # Check if projection rewrite can be done
    target_columns = [
        "order_id", "p_id", "order_no", "flight_id", "flight_no", "airline", "departure_city", 
        "arrival_city", "departure_airport", "arrival_airport", "departure_time", "arrival_time", 
        "gate", "seat_no", "cabin_class", "passenger_name", "passenger_phone", "passenger_id_no", 
        "amount", "status", "created_at"
    ]
    target_cols_set = {col.lower() for col in target_columns}
    
    # Identify which items in SELECT are from replaced sources or are unqualified
    items_to_rewrite = []
    can_rewrite_projection = True
    
    for item in select_items:
        is_replaced = references_aliases(item, legacy_lower)
        # Check if unqualified (no '.' at all)
        has_dot = False
        def check_dot(t):
            if t.ttype is sqlparse.tokens.Punctuation and t.value == '.':
                return True
            if hasattr(t, 'tokens') and t.tokens:
                return any(check_dot(sub) for sub in t.tokens)
            return False
        has_dot = check_dot(item)
        
        is_unqualified = not has_dot
        
        if is_replaced or is_unqualified:
            out_name = get_output_name(item)
            if out_name.lower() in target_cols_set:
                items_to_rewrite.append((item, out_name))
            else:
                # If it's replaced but not in target columns, we cannot rewrite the projection!
                if is_replaced:
                    can_rewrite_projection = False
                    break
                
    print(f"Can rewrite projection: {can_rewrite_projection}")
    
    if can_rewrite_projection:
        # We rewrite the select items!
        # Let's map original tokens/identifiers to new ones
        for item, out_name in items_to_rewrite:
            # Construct the new identifier token
            new_val = f"{target_alias}.{out_name}" if target_alias else out_name
            parsed_new = sqlparse.parse(new_val)[0]
            new_item_token = parsed_new.tokens[0]
            
            # We need to replace `item` with `new_item_token` in the parent token's tokens list
            # Find the parent of `item` (which could be Statement or IdentifierList)
            parent = item.parent
            if parent and hasattr(parent, 'tokens'):
                idx_in_parent = parent.tokens.index(item)
                parent.tokens[idx_in_parent] = new_item_token
                new_item_token.parent = parent
                
    # Rewrite other identifiers in the entire query (e.g. in WHERE clause)
    rewrite_identifiers(parsed, legacy_lower, target_alias)
    
    new_tokens = tokens[:from_idx + 1]
    if from_idx + 1 < len(tokens) and tokens[from_idx + 1].is_whitespace:
        new_tokens.append(tokens[from_idx + 1])
    else:
        new_tokens.append(sqlparse.sql.Token(sqlparse.tokens.Whitespace, " "))
        
    new_tokens.append(new_token)
    new_tokens.extend(middle_tokens)
    
    if end_idx > 0 and end_idx < len(tokens):
        pre_end_token = tokens[end_idx - 1]
        if pre_end_token.is_whitespace:
            new_tokens.append(pre_end_token)
            
    if end_idx < len(tokens):
        new_tokens.extend(tokens[end_idx:])
        
    parsed.tokens = new_tokens
    
    # Let's inspect the entire token tree of the modified parse
    def print_tokens(token, indent=""):
        name = type(token).__name__
        val = repr(token.value)
        ttype = getattr(token, "ttype", None)
        print(f"{indent}{name} ({ttype}): {val}")
        if hasattr(token, "tokens"):
            for sub in token.tokens:
                print_tokens(sub, indent + "  ")
                
    print("\n--- TOKEN TREE ---")
    print_tokens(parsed)
    
    return str(parsed)

sql = """
        SELECT
            o.order_id,
            o.p_id,
            b.amount,
            f.flight_no
        FROM ticket_order o
        JOIN passenger_info p ON p.p_id = o.p_id
        JOIN keep_this_table k ON k.id = o.k_id
        LEFT JOIN flight_info f ON f.flight_id = o.flight_id
        WHERE {where_clause}
"""

replaces = ["passenger_info", "flight_info"]
res = rewrite_query_source(sql, "view_ticket_report_detail", replaces)
print("RESULT:")
print(res)

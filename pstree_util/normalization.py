from collections import defaultdict, deque

def normalize_process_tree(tree):
    """
    Normalize the credentials (PIDs, SIDs, PGIDs) of a process tree.
    """
    # Step 1: Assign normalized PIDs
    pid_map = {}  # Maps original PIDs to normalized PIDs
    normalized_pid = 1

    # Breadth-first traversal to assign normalized PIDs
    queue = deque()
    queue.append(1)  # Start with the root process (init)

    while queue:
        original_pid = queue.popleft()
        pid_map[original_pid] = normalized_pid
        normalized_pid += 1

        # Add children to the queue
        for pid, process in tree.items():
            if process["ppid"] == original_pid:
                queue.append(pid)

    # Step 2: Assign normalized SIDs and PGIDs
    normalized_tree = {}
    for pid, process in tree.items():
        # Normalized PID
        normalized_pid = pid_map[pid]

        # Normalized SID: Derived from the session leader's normalized PID
        session_leader_pid = process["sid"]
        normalized_sid = pid_map[session_leader_pid]

        # Normalized PGID: Derived from the process group leader's normalized PID
        process_group_leader_pid = process["pgid"]
        normalized_pgid = pid_map[process_group_leader_pid]

        # Update the process with normalized credentials
        normalized_tree[normalized_pid] = {
            "ppid": pid_map[process["ppid"]],
            "sid": normalized_sid,
            "pgid": normalized_pgid,
            "comm": process["comm"],
        }

    return normalized_tree

if __name__ == '__main__':
  tree = {
    1: {"ppid": 0, "sid": 1, "pgid": 1, "comm": "init"},
    215: {"ppid": 1, "sid": 215, "pgid": 215, "comm": "bash"},
    300: {"ppid": 215, "sid": 300, "pgid": 300, "comm": "python3"},
    4: {"ppid": 215, "sid": 215, "pgid": 4, "comm": "python3"},
  }

  # Normalize the process tree
  normalized_tree = normalize_process_tree(tree)

  # Print the normalized tree
  print("Normalized Process Tree:")
  for pid, process in normalized_tree.items():
    print(f"PID: {pid}, PPID: {process['ppid']}, SID: {process['sid']}, PGID: {process['pgid']}, COMM: {process['comm']}")

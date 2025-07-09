def format_seconds(seconds: int) -> str:
    """
    Converts a given number of seconds into a human-readable time string.

    Parameters:
        seconds (int): The total number of seconds.

    Returns:
        str: A string representing the time in the format of hs ms s,
             where h, m, and s are hours, minutes, and seconds respectively.
             Only includes units with non-zero values.
    """
    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    seconds = round(remaining % 60)

    time_parts = []
    if hours > 0:
        time_parts.append(f"{int(hours)}h")
    if minutes > 0:
        time_parts.append(f"{int(minutes)}m")
    if seconds > 0:
        time_parts.append(str(seconds) + "s")

    if time_parts == []:
        return "0s"

    return " ".join(time_parts)

from dask.distributed import get_client


def check_client_active() -> bool:
    """
    Check whether a dask Client is active.
    :return: True/False whether a client is active or not
    """
    try:
        # This will succeed if a Dask client is running
        _ = get_client()
        return True
    except ValueError:
        # maybe this helps?
        # https://chatgpt.com/share/68daaca2-d128-8003-b038-b18a36620a08
        return False


def require_dask_client():
    client_active = check_client_active()
    if not client_active:
        raise NotImplementedError(
            "At the moment, using openEO-ML functionality only works in a distributed "
            "environment. At least run a LocalCluster. For best efficiency use "
            "dask.distributed.LocalCluster(processes=False)"
        )

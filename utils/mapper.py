import json


class ProblemMap(object):
    """ProblemSet encapsulating all the problems and concepts.

    """

    def __init__(self):
        """ Init ProblemMap Instance.

        """

        self.pid2pidx = {}
        self.pidx2pid = {}

        self.cid2cidx = {}
        self.cidx2cid = {}

        self.pidx2cidx = {}
        self.cidx2cname = {}
        self.num_concepts = None

    def __getitem__(self, pid):
        """ Retrieve pid's index.
        @param pid (str): problem id to look up.
        @return:
            index (int): index of word
        """
        return self.pid2pidx.get(pid)

    def __len__(self):
        """ Compute number of pid in ProblemMap.
        @return:
            len (int): number of pid in ProblemMap
        """
        return len(self.pid2pidx)

    def __contains__(self, pid: int):
        """ Check if pid is captured by ProblemSet.
        @param pid (int): pid to look up
        @return:
            contains (bool): whether word is contained
        """
        return pid in self.pid2pidx

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the ProblemSet.
        """
        raise ValueError('ProblemSet is readonly')

    def load_pid2pidx(self, path):
        with open(path, 'r') as f:
            temp = json.load(f)
            self.pid2pidx = {int(k): int(v) for k, v in temp.items()}
            self.pidx2pid = {v: k for k, v in self.pid2pidx.items()}

    def load_cid2cidx(self, path):
        with open(path, 'r') as f:
            temp = json.load(f)
            self.cid2cidx = {int(k): int(v) for k, v in temp.items()}
            self.cidx2cid = {v: k for k, v in temp.items()}
        self.num_concepts = len(self.cid2cidx)

    def load_pidx2cidx(self, path):
        with open(path, 'r') as f:
            temp = json.load(f)
            self.pidx2cidx = {int(k): int(v) for k, v in temp.items()}

    def load_cidx2cname(self, path):
        with open(path, 'r') as f:
            temp = json.load(f)
            self.cidx2cname = {int(k): str(v) for k, v in temp.items()}

    def pidx2pid(self, pidx: int):
        """ Return mapping of index to pid.
        @param pidx (int): problem index
        @return:
         pid (int): pid corresponding to index
        """
        return self.pidx2pid[pidx]

    def pid2pindices(self, pids):
        """ Convert list of pids or list of sequences of pids
        into list or list of list of indices.
        @param pids (list[str] or list[list[str]]): sequence(s) in pids
        @return problem indicex (list[int] or list[list[int]]): sequence(s) in indices
        """
        if type(pids[0]) == list:
            return [[self[pid] for pid in s] for s in pids]
        else:
            return [self[pid] for pid in pids]

    def indices2pid(self, pidxes):
        """ Convert list of pidxes or list of sequences of pidxes
        into list or list of list of pids.
        @param pidxes (list[str] or list[list[str]]): sequence(s) in indices
        @return problem ids (list[int] or list[list[int]]): sequence(s) in pids
        """
        if type(pidxes[0]) == list:
            return [[self.pidx2pid[pidx] for pidx in s] for s in pidxes]
        else:
            return [self.pidx2pid[pidx] for pidx in pidxes]

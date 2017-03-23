class SplitInfo(object):
    def __init__(self, left, right):
        assert len(left) != 0 and len(right) != 0, "You must split categorical attributes into two subset"
        self.left = [v for v in left]
        self.right = [v for v in right]
        if min(self.left) > min(self.right):
            temp = self.left
            self.left = self.right
            self.right = temp
        self._hasid()

    def __eq__(self, other):
        return self.hashid == other.hashid

    def _hasid(self):
        self.hashid = 0
        for val in sorted(self.left):
            self.hashid *= 10+val

    def __hash__(self):
        # This is not a good way to hash
        # but for now it's ok
        return hash(self.hashid)

    def __str__(self):
        return 'left %s right %s'%(self.left, self.right)

    def __repr__(self):
        return 'left %s right %s' % (self.left, self.right)


"""Declared measured-arrival dwell intention; no reset or body-state changes."""
class RouteDwell:
    def __init__(self,controls,actors):
        if type(controls) is not int or not 0<=controls<=300:raise ValueError('Dwell0..300 actual30Hz controls required')
        if controls and actors!=1:raise ValueError('Scalar speed dwell currently supports explicit single-actor narrow scenes')
        self.controls=controls;self.until=-1;self.anchor=None;self.events=[]
    def active(self,control):return control<self.until
    def arrived(self,control,reached,xy):
        if self.controls and bool(reached.any()):
            if self.active(control):raise ValueError('Cannot advance another waypoint during a dwell')
            self.anchor=xy.detach().clone();self.until=control+self.controls
            self.events.append({'start_control':control,'requested_end_control_exclusive':self.until,'measured_arrival_xy_m':self.anchor.cpu().tolist()})
    def targets(self):
        if self.anchor is None:raise ValueError('No actual arrival anchor')
        return self.anchor[:,None,:].expand(-1,10,2).clone()

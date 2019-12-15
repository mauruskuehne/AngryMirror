var NodeHelper = require('node_helper');
var zmq = require("zeromq/v5-compat");

module.exports = NodeHelper.create({
  start: function(){
    console.log(this.name + ' helper started ...');
    var that = this;
		try {
            
            sock = zmq.socket("sub");

			//sock.connect("tcp://127.0.0.1:5555");
			sock.connect("tcp://172.20.2.152:5555");
			console.log("Worker connected to port 5555");
			sock.subscribe('');
			sock.on("message", function(topic, message) {
			 console.log("work: %s", topic.toString());
			 that.sendSocketNotification('person', JSON.parse(topic.toString()));
			});
		} catch (error) {
			console.log(error);
		}
		
  }
});

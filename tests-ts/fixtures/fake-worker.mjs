import readline from "node:readline";
const send=(value)=>process.stdout.write(`${JSON.stringify(value)}\n`);
send({event:"ready",audio:true});
queueMicrotask(() => send({event:"model_ready",model:"fake"}));
for await (const line of readline.createInterface({input:process.stdin})) {
  const message=JSON.parse(line);
  const {id,command}=message;
  if(command==="start") { send({id,ok:true,event:"recording"}); send({event:"level",level:0.5}); }
  else if(command==="stop") send({id,ok:true,event:"transcript",text:"hello"});
  else if(command==="cancel") send({id,ok:true,event:"cancelled"});
  else if(command==="shutdown") { send({id,ok:true,event:"shutdown"}); break; }
  else send({id,ok:true,event:command});
}

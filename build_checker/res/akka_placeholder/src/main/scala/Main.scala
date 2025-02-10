import akka.actor.{Actor, ActorSystem, Props}

case class EchoMessage(message: String)

class EchoActor extends Actor {
  override def receive: Receive = {
    case msg @ EchoMessage(message) => context.system().actorSelection("/user/echo").tell(msg)
  }
}

object EchoActor {
  def apply(system: ActorSystem): ActorRef[EchoMessage] =
    system.actorOf(Props[EchoActor])
}

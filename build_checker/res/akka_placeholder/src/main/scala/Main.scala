import akka.actor.{Actor, ActorSystem, Props}

// Define the EchoActor
class EchoActor extends Actor {
  def receive: Receive = {
    case msg => sender() ! msg
  }
}

// Usage Example
object EchoTest extends App {
  val system = ActorSystem("EchoSystem")
  val echo = system.actorOf(Props[EchoActor](), "echo")

  echo ! "Test Message"
}